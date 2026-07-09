"""
generate_demo_checkpoint.py
============================
Creates a saved model checkpoint (.pt file) with INITIALISED weights.

Two modes:
  --mode random   : Save randomly-initialised weights (fastest, demo-only)
  --mode pretrain : Run a short self-supervised warm-up on synthetic data
                    (produces much better reconstruction quality)

The warm-up uses the synthetic data in data/ (run generate_demo_data.py first).
It trains for a small number of steps so reconstruction is plausible even
without the full CMRxRecon training set.

Output:
    checkpoints/demo_model.pt      ← model weights + config
    checkpoints/demo_model_info.txt ← human-readable summary

Usage:
    # Quick random init (pipeline test only):
    python generate_demo_checkpoint.py --mode random

    # 50-step warm-up on synthetic data (much better quality):
    python generate_demo_checkpoint.py --mode pretrain --steps 200

    # Custom architecture:
    python generate_demo_checkpoint.py --mode pretrain --n_cascades 3 --n_coils 10 --steps 100
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Project-local imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models.reconstruction_model import build_model, rss_combination
from transforms.fft_utils import ifft2c, fft2c
from datasets.dataset_loader import create_sampling_mask, apply_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_kspace(ks_path: str) -> torch.Tensor:
    """Load k-space from the synthetic .mat file."""
    import h5py
    with h5py.File(ks_path, "r") as f:
        if "kspace_data" in f:
            raw = f["kspace_data"][()]
        else:
            real = f["kspace_data_real"][()]
            imag = f["kspace_data_imag"][()]
            raw  = real + 1j * imag
    kspace = torch.from_numpy(raw.astype(np.complex64))
    # Normalise
    scale = kspace.abs().max().item()
    return kspace / (scale + 1e-8)


def _normalise_img(img: torch.Tensor) -> torch.Tensor:
    mn = img.amin(dim=(-2,-1), keepdim=True)
    mx = img.amax(dim=(-2,-1), keepdim=True)
    return (img - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------------------
# Mode 1 — Random initialisation
# ---------------------------------------------------------------------------

def save_random_checkpoint(
    n_coils:       int,
    n_cascades:    int,
    base_features: int,
    n_levels:      int,
    output_dir:    str,
):
    print("\n[generate_demo_checkpoint] Mode: RANDOM INIT")
    print("  Building model …")

    model = build_model(
        n_coils=n_coils,
        n_cascades=n_cascades,
        base_features=base_features,
        n_levels=n_levels,
    )
    model.summary()

    _save(model, output_dir, n_coils, n_cascades, base_features, n_levels,
          epoch=0, loss=None, mode="random_init")

    print("\n  ⚠️  Random weights: reconstruction quality will be poor.")
    print("     Run with --mode pretrain for better results.")


# ---------------------------------------------------------------------------
# Mode 2 — Short self-supervised warm-up
# ---------------------------------------------------------------------------

def pretrain_checkpoint(
    ks_path:       str,
    n_coils:       int,
    n_cascades:    int,
    base_features: int,
    n_levels:      int,
    acceleration:  int,
    steps:         int,
    lr:            float,
    output_dir:    str,
):
    print("\n[generate_demo_checkpoint] Mode: SELF-SUPERVISED PRETRAIN")
    device = _device()
    print(f"  Device: {device}")
    print(f"  Steps:  {steps}")

    # ── Load k-space ──────────────────────────────────────────────────────
    if not Path(ks_path).exists():
        print(f"\n  ⚠️  No k-space file found at: {ks_path}")
        print("     Run: python generate_demo_data.py  first.")
        print("     Falling back to random init.\n")
        save_random_checkpoint(n_coils, n_cascades, base_features,
                                n_levels, output_dir)
        return

    print(f"  Loading k-space: {ks_path} …")
    kspace_full = _load_kspace(ks_path)
    T, S, C, ky, kx = kspace_full.shape
    actual_coils = min(n_coils, C)
    kspace_full  = kspace_full[:, :, :actual_coils, :, :]
    print(f"  K-space shape: T={T} S={S} C={C}  →  using {actual_coils} coils")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(
        n_coils=actual_coils,
        n_cascades=n_cascades,
        base_features=base_features,
        n_levels=n_levels,
    ).to(device)
    model.summary()
    print()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps,
                                                       eta_min=lr * 0.05)
    criterion = nn.L1Loss()

    # ── Sampling mask (fixed for warm-up) ─────────────────────────────────
    mask = create_sampling_mask(ky, kx, acceleration=acceleration,
                                 mask_type="variable_density")
    mask_bc = mask.unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,ky,kx]

    # ── Collect all (frame, slice) pairs ──────────────────────────────────
    pairs = [(fi, si) for fi in range(T) for si in range(S)]

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    losses = []
    t0 = time.time()
    best_loss = float("inf")
    best_state = None

    print(f"  Warm-up training ({steps} steps) …")
    print(f"  {'Step':>6}  {'Loss':>10}  {'LR':>10}  {'Time':>8}")
    print(f"  {'----':>6}  {'----':>10}  {'--':>10}  {'----':>8}")

    for step in range(1, steps + 1):
        # Pick a random (frame, slice) pair
        fi, si = pairs[step % len(pairs)]
        ks_frame = kspace_full[fi, si].to(device)  # [C, ky, kx]

        # Apply undersampling mask
        ks_under = ks_frame * mask.to(device)      # [C, ky, kx]

        # Self-supervised target: RSS image from fully-sampled k-space
        with torch.no_grad():
            coil_imgs = ifft2c(ks_frame.unsqueeze(0))   # [1, C, H, W]
            target    = rss_combination(coil_imgs)       # [1, H, W]
            target_n  = _normalise_img(target)

        # Forward pass
        optimizer.zero_grad()
        recon  = model(ks_under.unsqueeze(0), mask_bc)   # [1, H, W]
        recon_n = _normalise_img(recon)
        loss   = criterion(recon_n, target_n)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if step % max(1, steps // 10) == 0 or step == 1:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            avg     = np.mean(losses[-20:]) if len(losses) >= 20 else np.mean(losses)
            print(f"  {step:>6}  {avg:>10.5f}  {lr_now:>10.2e}  {elapsed:>6.1f}s")

    elapsed_total = time.time() - t0
    print(f"\n  Warm-up complete in {elapsed_total:.1f}s  |  best loss: {best_loss:.5f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    _save(model, output_dir, actual_coils, n_cascades, base_features, n_levels,
          epoch=steps, loss=best_loss, mode="pretrained_warmup")

    print("\n  ✅  Pretrained checkpoint saved.")
    print(f"     Reconstruction quality: reasonable baseline from {steps}-step warm-up.")
    print("     For production quality: run train.py with the full CMRxRecon dataset.")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(
    model,
    output_dir:    str,
    n_coils:       int,
    n_cascades:    int,
    base_features: int,
    n_levels:      int,
    epoch:         int,
    loss,
    mode:          str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = out / "demo_model.pt"
    config = {
        "n_coils":       n_coils,
        "n_cascades":    n_cascades,
        "base_features": base_features,
        "n_levels":      n_levels,
        "mode":          mode,
    }

    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "loss":        loss,
            "config":      config,
        },
        ckpt_path,
    )

    # ── Human-readable info file ───────────────────────────────────────────
    info_path = out / "demo_model_info.txt"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(info_path, "w") as f:
        f.write("CardiacMRIReconNet — Model Checkpoint Info\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mode          : {mode}\n")
        f.write(f"Training steps: {epoch}\n")
        f.write(f"Final loss    : {loss}\n\n")
        f.write("Architecture:\n")
        f.write(f"  Model       : CardiacMRIReconNet (Unrolled U-Net)\n")
        f.write(f"  Cascades    : {n_cascades}\n")
        f.write(f"  Coils       : {n_coils}\n")
        f.write(f"  Base features: {base_features}\n")
        f.write(f"  U-Net levels: {n_levels}\n")
        f.write(f"  Parameters  : {n_params:,}\n\n")
        f.write("File:\n")
        f.write(f"  demo_model.pt\n\n")
        f.write("Usage:\n")
        f.write(f"  python inference.py --ks_path data/cine_sax_ks.mat \\\n")
        f.write(f"                      --ckpt_path checkpoints/demo_model.pt\n\n")
        f.write("  Or upload via the Streamlit UI: streamlit run ui/app.py\n")

    sz_mb = ckpt_path.stat().st_size / (1024**2)
    print(f"\n  Checkpoint : {ckpt_path}  ({sz_mb:.1f} MB)")
    print(f"  Info file  : {info_path}")
    print(f"  Parameters : {n_params:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate a demo model checkpoint for CardioRecon AI."
    )
    p.add_argument("--mode",          type=str, default="pretrain",
                   choices=["random", "pretrain"],
                   help="'random' = random init only. 'pretrain' = short warm-up training.")
    p.add_argument("--ks_path",       type=str, default="data/cine_sax_ks.mat")
    p.add_argument("--output_dir",    type=str, default="checkpoints")
    p.add_argument("--n_cascades",    type=int, default=5)
    p.add_argument("--n_coils",       type=int, default=10)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--n_levels",      type=int, default=3)
    p.add_argument("--acceleration",  type=int, default=4)
    p.add_argument("--steps",         type=int, default=200,
                   help="Number of warm-up training steps (pretrain mode only).")
    p.add_argument("--lr",            type=float, default=1e-3)
    args = p.parse_args()

    if args.mode == "random":
        save_random_checkpoint(
            n_coils=args.n_coils,
            n_cascades=args.n_cascades,
            base_features=args.base_features,
            n_levels=args.n_levels,
            output_dir=args.output_dir,
        )
    else:
        pretrain_checkpoint(
            ks_path=args.ks_path,
            n_coils=args.n_coils,
            n_cascades=args.n_cascades,
            base_features=args.base_features,
            n_levels=args.n_levels,
            acceleration=args.acceleration,
            steps=args.steps,
            lr=args.lr,
            output_dir=args.output_dir,
        )
