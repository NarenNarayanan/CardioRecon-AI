"""
Utils — Visualisation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Optional, Union, List
import imageio
import torch


def plot_kspace(
    kspace: Union[torch.Tensor, np.ndarray],
    title: str = "K-space Magnitude (log scale)",
    save_path: Optional[Union[str, Path]] = None,
    coil_idx: int = 0,
    frame_idx: int = 0,
    slice_idx: int = 0,
) -> plt.Figure:
    if isinstance(kspace, torch.Tensor):
        kspace = kspace.cpu().numpy()
    if kspace.ndim == 5:
        ks_2d = kspace[frame_idx, slice_idx, coil_idx]
    elif kspace.ndim == 3:
        ks_2d = kspace[coil_idx]
    else:
        ks_2d = kspace
    ks_log = np.log(np.abs(ks_2d) + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.imshow(ks_log, cmap="gray", interpolation="nearest")
    ax.set_title(title, color="white", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    return fig


def plot_reconstruction_comparison(
    zero_filled:   np.ndarray,
    reconstructed: np.ndarray,
    reference:     Optional[np.ndarray] = None,
    frame_idx:     int = 0,
    slice_idx:     int = 0,
    save_path:     Optional[Union[str, Path]] = None,
) -> plt.Figure:
    def _extract(arr, fi, si):
        if arr.ndim == 4:
            return arr[fi, si]       # [T, S, H, W] → [H, W]
        if arr.ndim == 3:
            return arr[fi]           # [T, H, W]    → [H, W]
        return arr                   # already [H, W]

    zf  = _extract(zero_filled,   frame_idx, slice_idx)
    rec = _extract(reconstructed, frame_idx, slice_idx)

    n_panels = 3 if reference is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                             facecolor="black")

    def _show(ax, img, ttl):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(ttl, color="white", fontsize=11, pad=6)
        ax.axis("off")

    _show(axes[0], _normalise(zf),  "Zero-filled (aliased)")
    _show(axes[1], _normalise(rec), "Reconstructed")

    if reference is not None:
        ref = _extract(reference, frame_idx, slice_idx)
        _show(axes[2], _normalise(ref), "Reference (ground truth)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    return fig


def save_cine_gif(
    frames:    np.ndarray,
    save_path: Union[str, Path],
    fps:       int = 15,
    loop:      int = 0,
) -> Path:
    save_path    = Path(save_path)
    frames_uint8 = (frames * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(str(save_path), frames_uint8, fps=fps, loop=loop, format="GIF")
    print(f"[save_cine_gif] Saved {len(frames)}-frame GIF → {save_path}")
    return save_path


def create_cine_animation(
    frames: np.ndarray,
    fps:    int = 15,
    title:  str = "Cardiac Cine MRI",
) -> animation.FuncAnimation:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.axis("off")
    fig.suptitle(title, color="white", fontsize=13)
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1,
                   animated=True, interpolation="bilinear")

    def _update(fi):
        im.set_data(frames[fi])
        return [im]

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames),
        interval=int(1000 / fps), blit=True,
    )
    plt.close(fig)
    return anim


def save_frame_png(frame: np.ndarray, save_path: Union[str, Path]) -> Path:
    save_path    = Path(save_path)
    frame_uint8  = (_normalise(frame) * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite(str(save_path), frame_uint8)
    return save_path


def save_all_frames(
    frames:     np.ndarray,
    output_dir: Union[str, Path],
    prefix:     str = "frame",
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = save_frame_png(frame, output_dir / f"{prefix}_{i:03d}.png")
        paths.append(p)
    print(f"[save_all_frames] Saved {len(paths)} frames → {output_dir}")
    return paths


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim_fn
    return float(ssim_fn(pred, target, data_range=1.0))


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def _normalise(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)
