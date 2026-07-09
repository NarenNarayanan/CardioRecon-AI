"""
Module 3 — Deep Reconstruction Model
======================================
Implements a simplified unrolled reconstruction network inspired by
PromptMR and VS-Net / E2E-VarNet architectures.

Architecture
------------

    Input: undersampled k-space  [B, C, ky, kx]  (B=batch, C=coils)
          ↓
    Cascade 1: [DataConsistency] → [U-Net denoiser]
          ↓
    Cascade 2: [DataConsistency] → [U-Net denoiser]
          ↓
    ...
    Cascade N: [DataConsistency] → [U-Net denoiser]
          ↓
    RSS coil combination  →  magnitude image  [B, y, x]

Each cascade refines the current k-space / image estimate.
Data consistency after the denoiser re-anchors predictions to measurements.

Complex tensors are handled by stacking real and imaginary channels
before passing to the U-Net, then unstacking afterward:
    complex [..., C, H, W]  →  real [..., 2*C, H, W]  →  complex [..., C, H, W]
"""

import torch
import torch.nn as nn

from transforms.fft_utils import fft2c, ifft2c
from models.unet import UNet, count_parameters
from reconstruction.data_consistency import SoftDataConsistency


# ---------------------------------------------------------------------------
# Complex ↔ real-channel conversion helpers
# ---------------------------------------------------------------------------

def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """
    Stack real and imaginary as channel pairs.

    [..., C, H, W]  →  [..., 2*C, H, W]
    """
    return torch.cat([x.real, x.imag], dim=-3)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Merge channel pairs back to complex.

    [..., 2*C, H, W]  →  [..., C, H, W]
    """
    c2 = x.shape[-3]
    assert c2 % 2 == 0, "Channel dimension must be even."
    c = c2 // 2
    return torch.complex(x[..., :c, :, :], x[..., c:, :, :])


# ---------------------------------------------------------------------------
# Single reconstruction cascade
# ---------------------------------------------------------------------------

class ReconCascade(nn.Module):
    """
    One unrolled reconstruction cascade:

        image_in  →  [U-Net denoiser]  →  [Data Consistency]  →  image_out

    The U-Net operates in image space on real+imag channel representation.
    Data consistency is applied in k-space after transforming the result.

    Args:
        n_coils:       Number of MRI receive coils.
        base_features: Base channel count for the U-Net (doubles per level).
        n_levels:      Depth of the U-Net (encoder/decoder levels).
    """

    def __init__(
        self,
        n_coils:       int = 10,
        base_features: int = 32,
        n_levels:      int = 3,
    ):
        super().__init__()

        # The U-Net sees 2 channels per coil (real + imag)
        self.unet = UNet(
            in_channels=2 * n_coils,
            out_channels=2 * n_coils,
            base_features=base_features,
            n_levels=n_levels,
        )

        self.dc = SoftDataConsistency(init_lambda=5.0)

    def forward(
        self,
        kspace_current:  torch.Tensor,
        kspace_measured: torch.Tensor,
        mask:            torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            kspace_current:  Current k-space estimate  [B, C, ky, kx] complex.
            kspace_measured: Original undersampled k-s  [B, C, ky, kx] complex.
            mask:            Sampling mask               [1, 1, ky, kx] float.

        Returns:
            kspace_out: Refined k-space estimate [B, C, ky, kx] complex.
        """
        # 1. Transform current k-space estimate to image space
        image = ifft2c(kspace_current)              # [B, C, H, W] complex

        # 2. Convert to real channels for U-Net
        image_real = complex_to_real(image)         # [B, 2C, H, W] real

        # 3. U-Net denoising step
        image_denoised_real = self.unet(image_real) # [B, 2C, H, W] real

        # 4. Convert back to complex
        image_denoised = real_to_complex(image_denoised_real)  # [B, C, H, W]

        # 5. Transform denoised image back to k-space
        kspace_denoised = fft2c(image_denoised)     # [B, C, ky, kx] complex

        # 6. Enforce data consistency with measured k-space
        kspace_out = self.dc(kspace_denoised, kspace_measured, mask)

        return kspace_out


# ---------------------------------------------------------------------------
# Full unrolled reconstruction network
# ---------------------------------------------------------------------------

class CardiacMRIReconNet(nn.Module):
    """
    Unrolled cardiac MRI reconstruction network.

    Stacks `n_cascades` ReconCascade modules to iteratively refine the
    k-space estimate from a zero-filled initialisation to a high-quality
    reconstruction.

    Pipeline:
        undersampled k-space
              ↓
        [Cascade 1]  →  data-consistent k-space₁
              ↓
        [Cascade 2]  →  data-consistent k-space₂
              ↓
             ...
        [Cascade N]  →  data-consistent k-spaceN
              ↓
        IFFT → coil images
              ↓
        RSS coil combination
              ↓
        magnitude image  [B, H, W]

    Args:
        n_coils:       Number of receive coils (default 10).
        n_cascades:    Number of unrolled cascades (default 5).
        base_features: U-Net base channels per cascade (default 32).
        n_levels:      U-Net depth per cascade (default 3).
    """

    def __init__(
        self,
        n_coils:       int = 10,
        n_cascades:    int = 5,
        base_features: int = 32,
        n_levels:      int = 3,
    ):
        super().__init__()
        self.n_coils    = n_coils
        self.n_cascades = n_cascades

        self.cascades = nn.ModuleList([
            ReconCascade(
                n_coils=n_coils,
                base_features=base_features,
                n_levels=n_levels,
            )
            for _ in range(n_cascades)
        ])

    def forward(
        self,
        kspace_under: torch.Tensor,
        mask:         torch.Tensor,
    ) -> torch.Tensor:
        """
        Full reconstruction forward pass.

        Args:
            kspace_under: Undersampled k-space [B, C, ky, kx] complex.
            mask:         Binary sampling mask  [1, 1, ky, kx] (float or bool).

        Returns:
            recon: Magnitude image  [B, H, W]  (root-sum-of-squares combined).
        """
        # Ensure mask is float for arithmetic in data consistency
        mask = mask.float().to(kspace_under.device)

        # Initialise with zero-filled estimate (= undersampled input)
        kspace_est = kspace_under

        # Iterative refinement through cascades
        for cascade in self.cascades:
            kspace_est = cascade(kspace_est, kspace_under, mask)

        # Transform final k-space to image space
        coil_images = ifft2c(kspace_est)             # [B, C, H, W] complex

        # Root Sum of Squares coil combination
        recon = rss_combination(coil_images)          # [B, H, W] real

        return recon

    def count_params(self) -> int:
        return count_parameters(self)

    def summary(self):
        n = self.count_params()
        print(f"CardiacMRIReconNet")
        print(f"  Cascades   : {self.n_cascades}")
        print(f"  Coils      : {self.n_coils}")
        print(f"  Parameters : {n:,}")
        print(f"  Per cascade: {n // self.n_cascades:,}")


# ---------------------------------------------------------------------------
# RSS coil combination
# ---------------------------------------------------------------------------

def rss_combination(coil_images: torch.Tensor) -> torch.Tensor:
    """
    Root Sum of Squares (RSS) coil combination.

    Combines multi-coil complex images into a single magnitude image by
    computing the square root of the sum of squared coil magnitudes.

    Formula:  RSS = sqrt( Σ_c  |image_c|² )

    Args:
        coil_images: Complex tensor [B, C, H, W]  where C = number of coils.

    Returns:
        rss: Real magnitude tensor [B, H, W].
    """
    return torch.sqrt((coil_images.abs() ** 2).sum(dim=1))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    n_coils:       int = 10,
    n_cascades:    int = 5,
    base_features: int = 32,
    n_levels:      int = 3,
) -> CardiacMRIReconNet:
    """
    Factory function — builds and returns a CardiacMRIReconNet.

    Example:
        >>> model = build_model(n_coils=8, n_cascades=3)
        >>> model.summary()
    """
    model = CardiacMRIReconNet(
        n_coils=n_coils,
        n_cascades=n_cascades,
        base_features=base_features,
        n_levels=n_levels,
    )
    return model
