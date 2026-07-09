"""
inference.py — Inference Pipeline for CardiacMRIReconNet
==========================================================
Runs the pre-trained reconstruction model on an input k-space file
and produces:
    • PNG frames for each cardiac phase
    • Animated GIF of the beating heart sequence
    • (Optionally) a side-by-side comparison plot

Usage:
    python inference.py \\
        --ks_path   data/cine_sax_ks.mat \\
        --ckpt_path checkpoints/best_model.pt \\
        --output_dir results/ \\
        --acceleration 4 \\
        --slice_idx 0 \\
        --fps 15 \\
        --compare

Or call reconstruct_from_file() directly from Python or the Streamlit UI.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

# ── Project-local imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from datasets.dataset_loader import load_kspace, create_sampling_mask, apply_mask
from transforms.fft_utils    import ifft2c, normalise_kspace
from models.reconstruction_model import build_model, rss_combination
from utils.coil_combination  import extract_cine_frames, normalise_volume
from utils.visualization     import (
    save_cine_gif,
    save_all_frames,
    plot_reconstruction_comparison,
    plot_kspace,
)


# ---------------------------------------------------------------------------
# Core reconstruction function (used by both CLI and Streamlit UI)
# ---------------------------------------------------------------------------

def reconstruct_from_file(
    ks_path:      str,
    ckpt_path:    Optional[str]  = None,
    acceleration: int            = 4,
    mask_type:    str            = "variable_density",
    n_coils:      Optional[int]  = None,
    n_cascades:   int            = 5,
    base_features:int            = 32,
    n_levels:     int            = 3,
    device:       Optional[str]  = None,
    progress_cb   = None,
) -> Dict[str, Any]:
    """
    Full inference pipeline: k-space file → reconstructed cine frames.

    Args:
        ks_path:       Path to .mat k-space file.
        ckpt_path:     Path to trained model checkpoint (.pt).
                       If None, uses a randomly-initialised model (demo mode).
        acceleration:  Undersampling acceleration factor.
        mask_type:     Sampling mask type ('variable_density', 'cartesian', 'random').
        n_coils:       Number of coils to use (None = use all in file).
        n_cascades:    Number of unrolled cascades in the model.
        base_features: U-Net base channel count.
        n_levels:      U-Net depth.
        device:        Device string ('cpu', 'cuda', 'mps', or None = auto-detect).
        progress_cb:   Optional callable(step: int, total: int, msg: str)
                       for progress reporting (used by Streamlit).

    Returns:
        Dictionary with keys:
            'frames_recon'    : np.ndarray [T, H, W]  reconstructed frames
            'frames_zerofill' : np.ndarray [T, H, W]  zero-filled frames
            'kspace_under'    : torch.Tensor           undersampled k-space
            'mask'            : torch.Tensor           sampling mask
            'n_coils'         : int
            'shape'           : tuple  (T, S, C, ky, kx)
    """
    def _cb(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)
        else:
            pct = int(step / total * 100)
            print(f"  [{pct:3d}%] {msg}")

    _cb(0, 6, "Loading k-space data …")

    # ── 1. Load and normalise k-space ──────────────────────────────────────
    kspace_full = load_kspace(ks_path)
    kspace_full, scale = _normalise_ks(kspace_full)
    T, S, C, ky, kx = kspace_full.shape

    # Coil selection
    actual_n_coils = min(n_coils, C) if n_coils else C
    kspace_full = kspace_full[:, :, :actual_n_coils, :, :]

    _cb(1, 6, f"K-space loaded: T={T} S={S} C={C}  →  using {actual_n_coils} coils")

    # ── 2. Resolve device ──────────────────────────────────────────────────
    if device is None:
        device_obj = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device_obj = torch.device(device)

    _cb(2, 6, f"Device: {device_obj}")

    # ── 3. Create undersampling mask ───────────────────────────────────────
    mask = create_sampling_mask(
        ky, kx,
        acceleration=acceleration,
        mask_type=mask_type,
    )

    # ── 4. Build / load model ──────────────────────────────────────────────
    _cb(3, 6, "Building reconstruction model …")
    model = build_model(
        n_coils=actual_n_coils,
        n_cascades=n_cascades,
        base_features=base_features,
        n_levels=n_levels,
    ).to(device_obj)

    if ckpt_path and Path(ckpt_path).exists():
        _load_checkpoint(model, ckpt_path, device_obj)
        _cb(3, 6, f"Checkpoint loaded: {ckpt_path}")
    else:
        _cb(3, 6, "No checkpoint found — running in demo mode (random weights)")

    model.eval()

    # ── 5. Reconstruct each slice ──────────────────────────────────────────
    _cb(4, 6, f"Reconstructing {T} frames × {S} slices …")

    recon_volume   = torch.zeros(T, S, ky, kx)    # [T, S, H, W]
    zerofill_volume= torch.zeros(T, S, ky, kx)

    mask_bc = mask.unsqueeze(0).unsqueeze(0).float().to(device_obj)  # [1,1,ky,kx]

    with torch.no_grad():
        for si in range(S):
            ks_slice = kspace_full[:, si, :, :, :]   # [T, C, ky, kx]

            for fi in range(T):
                ks_frame  = ks_slice[fi]              # [C, ky, kx]
                ks_under  = ks_frame.to(device_obj) * mask.to(device_obj)  # [C, ky, kx]

                # Zero-filled baseline (RSS of IFFT of undersampled k-space)
                zf_imgs   = ifft2c(ks_under.unsqueeze(0))   # [1, C, H, W]
                zf_recon  = rss_combination(zf_imgs)[0]     # [H, W]

                # Model reconstruction
                ks_batch  = ks_under.unsqueeze(0)           # [1, C, ky, kx]
                recon     = model(ks_batch, mask_bc)[0]     # [H, W]

                recon_volume[fi, si]    = recon.cpu()
                zerofill_volume[fi, si] = zf_imgs[0, 0].abs().cpu()

    # ── 6. Post-process ────────────────────────────────────────────────────
    _cb(5, 6, "Post-processing and extracting cine frames …")

    recon_norm   = normalise_volume(recon_volume).numpy()
    zerofill_norm= normalise_volume(zerofill_volume).numpy()

    _cb(6, 6, "Reconstruction complete ✓")

    return {
        "frames_recon":     recon_norm,       # [T, S, H, W] float32
        "frames_zerofill":  zerofill_norm,    # [T, S, H, W] float32
        "kspace_under":     None,             # not returned to save memory
        "mask":             mask.numpy(),
        "n_coils":          actual_n_coils,
        "shape":            (T, S, actual_n_coils, ky, kx),
    }


# ---------------------------------------------------------------------------
# Save results helper
# ---------------------------------------------------------------------------

def save_results(
    results:    Dict[str, Any],
    output_dir: str,
    slice_idx:  int = 0,
    fps:        int = 15,
    save_frames:bool = True,
    compare:    bool = False,
) -> Dict[str, Path]:
    """
    Save reconstruction outputs to disk.

    Returns dict of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    frames_recon    = results["frames_recon"][:, slice_idx, :, :]    # [T, H, W]
    frames_zerofill = results["frames_zerofill"][:, slice_idx, :, :] # [T, H, W]

    # GIF — reconstructed
    gif_path = output_dir / "cine_recon.gif"
    save_cine_gif(frames_recon, gif_path, fps=fps)
    saved["gif_recon"] = gif_path

    # GIF — zero-filled
    gif_zf_path = output_dir / "cine_zerofill.gif"
    save_cine_gif(frames_zerofill, gif_zf_path, fps=fps)
    saved["gif_zerofill"] = gif_zf_path

    # Individual PNG frames
    if save_frames:
        frame_dir = output_dir / "frames"
        save_all_frames(frames_recon, frame_dir, prefix="recon")
        saved["frames_dir"] = frame_dir

    # Comparison plot
    if compare:
        T  = frames_recon.shape[0]
        mid = T // 2
        fig = plot_reconstruction_comparison(
            zero_filled=frames_zerofill,
            reconstructed=frames_recon,
            frame_idx=mid,
        )
        cmp_path = output_dir / "comparison.png"
        fig.savefig(cmp_path, dpi=150, bbox_inches="tight", facecolor="black")
        import matplotlib.pyplot as plt
        plt.close(fig)
        saved["comparison"] = cmp_path

    print(f"\n[inference] Results saved to: {output_dir}")
    for k, v in saved.items():
        print(f"  {k}: {v}")

    return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_ks(ks: torch.Tensor) -> Tuple[torch.Tensor, float]:
    scale = ks.abs().max().item()
    return ks / (scale + 1e-8), scale


def _load_checkpoint(
    model:    torch.nn.Module,
    ckpt_path: str,
    device:   torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        epoch = ckpt.get("epoch", "?")
        loss  = ckpt.get("loss",  "?")
        print(f"[inference] Checkpoint: epoch={epoch}  loss={loss}")
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="CardiacMRI Inference")
    p.add_argument("--ks_path",       type=str, required=True,
                   help="Path to cine_sax_ks.mat")
    p.add_argument("--ckpt_path",     type=str, default=None,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--output_dir",    type=str, default="results")
    p.add_argument("--acceleration",  type=int, default=4)
    p.add_argument("--mask_type",     type=str, default="variable_density",
                   choices=["variable_density", "cartesian", "random"])
    p.add_argument("--n_coils",       type=int, default=None)
    p.add_argument("--n_cascades",    type=int, default=5)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--n_levels",      type=int, default=3)
    p.add_argument("--slice_idx",     type=int, default=0)
    p.add_argument("--fps",           type=int, default=15)
    p.add_argument("--compare",       action="store_true",
                   help="Save side-by-side comparison plot")
    p.add_argument("--no_frames",     action="store_true",
                   help="Skip saving individual PNG frames")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"\n{'='*60}")
    print(f"  CardiacMRIReconNet — Inference")
    print(f"{'='*60}")
    print(f"  K-space   : {args.ks_path}")
    print(f"  Checkpoint: {args.ckpt_path or 'demo mode (random weights)'}")
    print(f"  Accel     : {args.acceleration}x  ({args.mask_type})")
    print(f"{'='*60}\n")

    t0 = time.time()

    results = reconstruct_from_file(
        ks_path=args.ks_path,
        ckpt_path=args.ckpt_path,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        n_coils=args.n_coils,
        n_cascades=args.n_cascades,
        base_features=args.base_features,
        n_levels=args.n_levels,
    )

    save_results(
        results,
        output_dir=args.output_dir,
        slice_idx=args.slice_idx,
        fps=args.fps,
        save_frames=not args.no_frames,
        compare=args.compare,
    )

    elapsed = time.time() - t0
    T, S, C, ky, kx = results["shape"]
    print(f"\n[inference] Done in {elapsed:.1f}s  "
          f"(reconstructed {T} frames × {S} slices, {ky}×{kx} px)")

