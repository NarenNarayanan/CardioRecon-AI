"""
Module 4 — Data Consistency Layer
===================================
Enforces physics-based consistency between the network prediction and
the actually measured k-space lines.

Mathematical formulation
------------------------
After the CNN denoiser updates the image estimate x̂, we transform it
back to k-space:

    k̂ = F(x̂)

Then we enforce data consistency by replacing the predicted values at
measured locations with the true measured values:

    k_DC[mask]  = k_measured[mask]          ← keep real measurements
    k_DC[~mask] = k̂[~mask]                  ← use network prediction elsewhere

Alternatively a soft/learnable version:

    k_DC = mask * k_measured + λ * (1-mask) * k̂ + (1-mask)(1-λ) * k̂

where λ ∈ [0,1] is a learnable trade-off parameter (soft DC).
"""

import torch
import torch.nn as nn
from transforms.fft_utils import fft2c, ifft2c


# ---------------------------------------------------------------------------
# Hard data consistency (standard)
# ---------------------------------------------------------------------------

def apply_data_consistency(
    kspace_pred:     torch.Tensor,
    kspace_measured: torch.Tensor,
    mask:            torch.Tensor,
) -> torch.Tensor:
    """
    Hard data consistency:  fully replace predicted k-space at measured locs.

    k_out = mask ⊙ k_measured + (1 − mask) ⊙ k_pred

    Args:
        kspace_pred:     Predicted k-space from the current cascade.
                         Shape: [..., ky, kx]  (complex).
        kspace_measured: Original undersampled measurements (zero at gaps).
                         Shape: [..., ky, kx]  (complex).
        mask:            Binary sampling mask  (1 = measured, 0 = missing).
                         Shape broadcastable to kspace_pred, e.g. [1,1,1,ky,kx].

    Returns:
        kspace_dc: Data-consistent k-space of same shape as kspace_pred.
    """
    mask = mask.to(kspace_pred.device)

    kspace_dc = mask * kspace_measured + (1.0 - mask) * kspace_pred
    return kspace_dc


def replace_kspace_values(
    kspace_pred:     torch.Tensor,
    kspace_measured: torch.Tensor,
    mask:            torch.Tensor,
) -> torch.Tensor:
    """
    Alias for apply_data_consistency with explicit 'replace' semantics.
    Measured k-space lines completely overwrite the network's prediction.
    """
    return apply_data_consistency(kspace_pred, kspace_measured, mask)


# ---------------------------------------------------------------------------
# Soft / learnable data consistency module
# ---------------------------------------------------------------------------

class SoftDataConsistency(nn.Module):
    """
    Learnable soft data consistency layer.

    Introduces a per-cascade trainable scalar λ (lambda) that controls
    how strongly measured k-space overrides the network prediction:

        k_out = mask ⊙ [λ·k_measured + (1−λ)·k_pred]
              + (1−mask) ⊙ k_pred

    When λ → 1  →  hard data consistency (full replacement).
    When λ → 0  →  no enforcement (pure network prediction).

    λ is initialised to 1.0 (hard DC) and constrained to [0, 1] via sigmoid.

    Args:
        init_lambda: Initial value of λ before sigmoid projection (default 5.0
                     → sigmoid(5) ≈ 0.993, very close to hard DC).
    """

    def __init__(self, init_lambda: float = 5.0):
        super().__init__()
        # Raw (unconstrained) parameter; projected to [0,1] via sigmoid
        self._lambda_raw = nn.Parameter(torch.tensor(init_lambda))

    @property
    def lambda_val(self) -> torch.Tensor:
        """λ in [0, 1] — controls DC strength."""
        return torch.sigmoid(self._lambda_raw)

    def forward(
        self,
        kspace_pred:     torch.Tensor,
        kspace_measured: torch.Tensor,
        mask:            torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply soft data consistency.

        Args:
            kspace_pred:     Predicted k-space  [..., ky, kx] (complex).
            kspace_measured: Measured  k-space  [..., ky, kx] (complex).
            mask:            Binary mask, broadcastable to kspace_pred.

        Returns:
            kspace_dc: Data-consistent k-space.
        """
        lam  = self.lambda_val
        mask = mask.to(kspace_pred.device)

        # At measured locations: blend measurement and prediction
        kspace_dc = (
            mask       * (lam * kspace_measured + (1.0 - lam) * kspace_pred)
            + (1.0 - mask) * kspace_pred
        )
        return kspace_dc


# ---------------------------------------------------------------------------
# Image-space data consistency (for reference)
# ---------------------------------------------------------------------------

class ImageSpaceDataConsistency(nn.Module):
    """
    Data consistency applied in image space.

    Steps:
        1. Forward FFT the current image estimate  → k_pred
        2. Replace measured k-space lines           → k_dc
        3. Inverse FFT back to image space         → x_dc

    This module is a convenience wrapper that combines the FFT utilities
    with data consistency in a single differentiable block.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_pred:      torch.Tensor,
        kspace_measured: torch.Tensor,
        mask:            torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_pred:      Current image estimate [..., y, x] (complex).
            kspace_measured: Undersampled k-space   [..., ky, kx] (complex).
            mask:            Binary mask, broadcastable to kspace_measured.

        Returns:
            image_dc: Data-consistent image estimate [..., y, x] (complex).
        """
        # Transform prediction to k-space
        kspace_pred = fft2c(image_pred)

        # Enforce consistency in k-space
        kspace_dc = apply_data_consistency(kspace_pred, kspace_measured, mask)

        # Transform back to image space
        image_dc = ifft2c(kspace_dc)
        return image_dc
