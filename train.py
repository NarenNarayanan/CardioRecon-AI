"""
train.py — Training Script for CardiacMRIReconNet
===================================================
Trains the unrolled reconstruction network on cardiac MRI k-space data.

Usage:
    python train.py --ks_path data/cine_sax_ks.mat \\
                    --n_cascades 5 \\
                    --n_coils 10 \\
                    --epochs 50 \\
                    --batch_size 1 \\
                    --lr 1e-4 \\
                    --acceleration 4 \\
                    --output_dir checkpoints/

Training strategy:
    • Input:  Simulated undersampled k-space (using variable density mask)
    • Target: Zero-filled RSS image (used as self-supervised target)
              OR fully-sampled RSS image if full k-space is provided
    • Loss:   L1 + SSIM composite loss
    • Optimiser: AdamW with cosine annealing LR schedule

Note:
    For a real training setup, provide matched (undersampled, ground-truth)
    pairs. Here we demonstrate the pipeline using simulated undersampling
    of provided k-space data.
"""

"""
train.py — Training Script for CardiacMRIReconNet
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from datasets.dataset_loader import load_kspace, create_sampling_mask, apply_mask
from transforms.fft_utils    import ifft2c
from models.reconstruction_model import build_model, rss_combination


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        return 0.7 * self.l1(pred, target) + 0.3 * self.l2(pred, target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_image(img: torch.Tensor) -> torch.Tensor:
    mn = img.amin(dim=(-2, -1), keepdim=True)
    mx = img.amax(dim=(-2, -1), keepdim=True)
    return (img - mn) / (mx - mn + 1e-8)


def normalise_kspace_local(ks):
    scale = ks.abs().max().item()
    return ks / (scale + 1e-8), scale


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  CardiacMRIReconNet — Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  K-space file: {args.ks_path}")
    print(f"  Cascades    : {args.n_cascades}")
    print(f"  Coils       : {args.n_coils}")
    print(f"  Accel       : {args.acceleration}x")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"{'='*60}\n")

    # ── Load k-space ──────────────────────────────────────────────────────
    print("[train] Loading k-space data ...")
    kspace_full = load_kspace(args.ks_path, max_coils=args.n_coils)
    kspace_full, _ = normalise_kspace_local(kspace_full)
    T, S, C, ky, kx = kspace_full.shape
    print(f"[train] k-space: T={T} S={S} C={C}  using {C} coils\n")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(
        n_coils=C,
        n_cascades=args.n_cascades,
        base_features=args.base_features,
        n_levels=args.n_levels,
    ).to(device)
    model.summary()
    print()

    # ── Create mask ONCE — reuse every iteration ──────────────────────────
    print("[train] Creating sampling mask (once) ...")
    mask = create_sampling_mask(
        ky=ky, kx=kx,
        acceleration=args.acceleration,
        mask_type="variable_density",
    )
    mask_bc = mask.unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,ky,kx]
    print()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = ReconLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-apply mask to all data ─────────────────────────────────────────
    # Shape: [T, S, C, ky, kx] — apply mask once, store result
    print("[train] Pre-applying mask to entire dataset ...")
    mask_5d = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,ky,kx]
    kspace_under_all = kspace_full * mask_5d               # [T,S,C,ky,kx]

    # Pre-compute all targets (RSS of full k-space) — avoids recomputing each step
    print("[train] Pre-computing reconstruction targets ...")
    with torch.no_grad():
        targets = torch.zeros(T, S, ky, kx)
        for fi in range(T):
            for si in range(S):
                coil_imgs = ifft2c(kspace_full[fi, si].unsqueeze(0))  # [1,C,H,W]
                targets[fi, si] = rss_combination(coil_imgs)[0]       # [H,W]
    print(f"[train] Targets shape: {targets.shape}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    pairs = [(fi, si) for fi in range(T) for si in range(S)]

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for fi, si in pairs:
            optimizer.zero_grad()

            ks_under = kspace_under_all[fi, si].unsqueeze(0).to(device)  # [1,C,ky,kx]
            target   = targets[fi, si].unsqueeze(0).to(device)           # [1,H,W]

            recon   = model(ks_under, mask_bc)   # [1,H,W]
            loss    = criterion(normalise_image(recon), normalise_image(target))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.5f}  lr={lr_now:.2e}  time={elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "loss":        best_loss,
                    "config": {
                        "n_coils":       C,
                        "n_cascades":    args.n_cascades,
                        "base_features": args.base_features,
                        "n_levels":      args.n_levels,
                    },
                },
                output_dir / "best_model.pt",
            )
            print(f"  ↳ Best model saved  (loss={best_loss:.5f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       output_dir / f"checkpoint_epoch{epoch:03d}.pt")

    print(f"\n[train] Done. Best loss: {best_loss:.5f}")
    print(f"[train] Checkpoint: {output_dir}/best_model.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ks_path",       type=str,   required=True)
    p.add_argument("--n_cascades",    type=int,   default=5)
    p.add_argument("--n_coils",       type=int,   default=10)
    p.add_argument("--base_features", type=int,   default=32)
    p.add_argument("--n_levels",      type=int,   default=3)
    p.add_argument("--acceleration",  type=int,   default=4)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--output_dir",    type=str,   default="checkpoints")
    p.add_argument("--save_every",    type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

