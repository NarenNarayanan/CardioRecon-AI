"""
train_deepssl.py — DeepSSL Training Pipeline
=============================================
Training script for the full DeepSSL paper implementation.

From paper (Section III-E):
    "Our DeepSSL is trained for 50 epochs with the Adam optimizer.
     Initial learning rate: 0.001 with exponential decay of 0.99.
     Batch size: 64.
     Typical training: ~50 hours on Nvidia Tesla T4 (16GB)."

Key differences from our simplified CardioRecon AI training:

    Paper (DeepSSL):                    Our implementation:
    ─────────────────────────────────   ──────────────────────────────────
    2D k-t slices [B, 2C, PE, T]        2D spatial images [B, C, H, W]
    1D convolutions (temporal+spatial)   2D U-Net convolutions
    Multi-phase loss (at every cascade)  Single loss (final output only)
    M × N_subjects training samples      T × S training samples
    Adam lr=0.001, decay=0.99            AdamW lr=0.0001, cosine annealing
    L2 loss only                         L1 + MSE composite loss
    ESPIRiT coil combination             Root Sum of Squares
    50 hours on Tesla T4                 45 min on RTX A4000
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.separable_dataset_generator import SeparableKtDataset
from models.deepssl_network import DeepSSL, DeepSSLLoss


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_deepssl(args):
    """
    Full DeepSSL training pipeline following the paper exactly.

    Training strategy:
        1. Build separable dataset (2D k-t slices from 3D k-space)
        2. Train DeepSSL with multi-phase L2 loss
        3. Adam optimiser with exponential LR decay
        4. Evaluate on validation set (RLNE, PSNR, SSIM)

    From Table II of paper (computational efficiency):
        DeepSSL memory usage (BS=1): ~1GB GPU
        Competing methods (BS=1):   4~10GB GPU
        DeepSSL parameters: 564,520
        → Uses 1D convolutions instead of 2D/3D → massive efficiency gain
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  DeepSSL Training — Deep Separable Spatiotemporal Learning")
    print(f"{'='*65}")
    print(f"  Paper: Wang et al., IEEE TBME 2025")
    print(f"  Device:       {device}")
    print(f"  K phases:     {args.K}")
    print(f"  Acceleration: {args.acceleration}x")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"{'='*65}\n")

    # ── Step 1: Build separable dataset ───────────────────────────────────
    print("[train] Building separable k-t dataset ...")
    print("[train] For each subject, 1D FE IFFT creates M independent")
    print("[train] 2D k-t slices → massive data augmentation.\n")

    train_dataset = SeparableKtDataset(
        ks_files     = args.train_files,
        acceleration = args.acceleration,
        calib_lines  = args.calib_lines,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    n_subjects = len(args.train_files)
    n_samples  = len(train_dataset)
    print(f"\n[train] Dataset summary:")
    print(f"  Subjects:          {n_subjects}")
    print(f"  Samples (2D k-t):  {n_samples}")
    print(f"  Augmentation:      {n_samples // n_subjects}× per subject")
    print(f"  (Direct learning would have: {n_subjects} samples)")
    print()

    # ── Step 2: Build model ────────────────────────────────────────────────
    model = DeepSSL(
        n_channels = args.n_channels,
        n_filters  = 48,          # fixed at 48 as in paper
        K          = args.K,      # 10 unrolled phases
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] DeepSSL model:")
    print(f"  Parameters: {n_params:,}")
    print(f"  (Paper reports 564,520 parameters)")
    print(f"  Convolution type: 1D only (temporal + spatial separately)")
    print()

    # ── Step 3: Optimiser — Adam with exponential decay ───────────────────
    # Exact specification from paper Section III-E
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Exponential decay: lr_t = lr_0 × 0.99^t
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ── Step 4: Loss function — multi-phase L2 ────────────────────────────
    # Loss computed at EVERY phase (not just final) — key difference from
    # standard training
    criterion = DeepSSLLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    # ── Step 5: Training loop ──────────────────────────────────────────────
    print(f"[train] Starting training for {args.epochs} epochs ...")
    print(f"[train] Expected time: ~50 hours on Tesla T4 (paper)")
    print(f"[train] Loss computed at all {args.K} phases (multi-phase supervision)\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (input_2d, target_2d) in enumerate(train_loader):
            # input_2d:  [B, 2C, PE, T]  — undersampled 2D k-t slice
            # target_2d: [B, PE, T]       — fully-sampled 2D image (ground truth)

            input_2d  = input_2d.to(device)
            target_2d = target_2d.to(device)

            # Create sampling mask (same for all items in batch)
            # In practice, mask is pre-computed per subject
            _, C2, PE, T = input_2d.shape
            mask = (input_2d[:1, 0].abs() > 1e-6).float()  # detect measured positions

            optimizer.zero_grad()

            # Forward pass — returns outputs from ALL K phases
            # This is the key: intermediate reconstructions are supervised
            phase_outputs = model(input_2d, mask)

            # Multi-phase loss (Eq. 10 from paper):
            # L = (1/KC) Σ_{k=1}^{K} ||X_ref - X^(k)||²₂
            loss = criterion(phase_outputs, target_2d)

            # Backward pass
            loss.backward()

            # Gradient clipping (standard practice)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()  # exponential LR decay

        avg_loss = np.mean(epoch_losses)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.6f}  "
              f"lr={lr_now:.4e}  "
              f"time={elapsed:.1f}s  "
              f"samples/s={len(train_dataset)/elapsed:.1f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "loss":        best_loss,
                    "config": {
                        "n_channels":  args.n_channels,
                        "n_filters":   48,
                        "K":           args.K,
                        "method":      "DeepSSL",
                        "paper":       "Wang et al., IEEE TBME 2025",
                    },
                },
                output_dir / "deepssl_best_model.pt",
            )
            print(f"  ↳ Best model saved (loss={best_loss:.6f})")

    print(f"\n[train] Training complete.")
    print(f"[train] Best loss: {best_loss:.6f}")
    print(f"[train] Checkpoint: {output_dir}/deepssl_best_model.pt")


# ---------------------------------------------------------------------------
# Evaluation metrics from paper
# ---------------------------------------------------------------------------

def compute_rlne(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Relative L2 Norm Error (RLNE) — primary metric used in paper.

        RLNE = ||pred - target||₂ / ||target||₂

    Lower is better. Paper reports RLNE in range 0.06-0.08 for
    state-of-the-art methods.
    """
    return (torch.norm(pred - target) / torch.norm(target)).item()


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

        PSNR = 20 × log10(MAX / RMSE)

    Higher is better. Paper reports PSNR ~38-42 dB for DeepSSL.
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    max_val = target.max()
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).

        SSIM = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    Higher is better. Range [0, 1]. Paper reports SSIM ~0.93-0.96 for DeepSSL.
    """
    from skimage.metrics import structural_similarity
    return float(structural_similarity(pred, target, data_range=target.max()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class Args:
    """Default arguments matching paper configuration."""
    train_files  = ["data/cine_sax_ks.mat"]
    output_dir   = "checkpoints_deepssl"
    K            = 10       # 10 unrolled phases (paper)
    n_channels   = 2        # real + imaginary
    acceleration = 6        # AF=6 (paper uses 6 as primary experiment)
    calib_lines  = 24       # ACS calibration lines
    epochs       = 50       # 50 epochs (paper)
    batch_size   = 64       # batch size 64 (paper)
    lr           = 0.001    # initial LR 0.001 (paper)


if __name__ == "__main__":
    args = Args()
    train_deepssl(args)