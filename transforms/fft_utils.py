"""
Module 2 — Hybrid Space Transformation
=======================================
Provides centred 2-D FFT utilities for converting between k-space
(frequency domain) and image space (spatial domain).

All operations use torch.fft for GPU-compatible, differentiable transforms.
Centring follows the MRI convention:  DC component at the array centre.

Tensor convention throughout:
    [..., ky, kx]   — k-space (last two dims are spatial frequencies)
    [..., y,  x ]   — image  space (last two dims are image pixels)
"""

import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shift utilities
# ---------------------------------------------------------------------------

def fftshift(x: torch.Tensor, dims: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Shift the zero-frequency component to the centre of the spectrum.

    Equivalent to numpy.fft.fftshift.

    Args:
        x:    Input tensor (any dtype, including complex).
        dims: Dimensions to shift. Defaults to the last two dims (-2, -1).

    Returns:
        Shifted tensor of the same shape and dtype.
    """
    if dims is None:
        dims = (-2, -1)
    shifts = [x.shape[d] // 2 for d in dims]
    return torch.roll(x, shifts, dims)


def ifftshift(x: torch.Tensor, dims: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Inverse of fftshift — shifts zero-frequency component back to corner.

    Args:
        x:    Input tensor.
        dims: Dimensions to shift. Defaults to (-2, -1).

    Returns:
        Shifted tensor of the same shape and dtype.
    """
    if dims is None:
        dims = (-2, -1)
    shifts = [-(x.shape[d] // 2) for d in dims]
    return torch.roll(x, shifts, dims)


# ---------------------------------------------------------------------------
# Centred 2-D FFT / IFFT
# ---------------------------------------------------------------------------

def fft2c(image: torch.Tensor) -> torch.Tensor:
    """
    Centred 2-D Fast Fourier Transform:  image space  →  k-space.

    Steps:
        1. ifftshift  (move DC to corner for torch.fft convention)
        2. fft2       (frequency transform)
        3. fftshift   (move DC back to centre)
        4. Normalise  (orthonormal convention: 1/√N)

    Args:
        image: Complex tensor with shape [..., y, x].

    Returns:
        kspace: Complex tensor with shape [..., ky, kx].

    Example:
        >>> img = torch.randn(1, 1, 10, 192, 192, dtype=torch.complex64)
        >>> ks  = fft2c(img)   # shape (1, 1, 10, 192, 192)
    """
    # Move DC component to corner (required by torch.fft.fft2)
    x = ifftshift(image, dims=(-2, -1))

    # Orthonormal 2-D DFT
    x = torch.fft.fft2(x, norm="ortho")

    # Shift DC back to centre
    x = fftshift(x, dims=(-2, -1))
    return x


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """
    Centred 2-D Inverse Fast Fourier Transform:  k-space  →  image space.

    Steps (mirror of fft2c):
        1. ifftshift  (move DC to corner)
        2. ifft2      (inverse frequency transform)
        3. fftshift   (move DC back to centre)
        4. Normalise  (orthonormal convention)

    Args:
        kspace: Complex tensor with shape [..., ky, kx].

    Returns:
        image: Complex tensor with shape [..., y, x].
    """
    x = ifftshift(kspace, dims=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    x = fftshift(x, dims=(-2, -1))
    return x


# ---------------------------------------------------------------------------
# Batch helpers used by the reconstruction cascade
# ---------------------------------------------------------------------------

def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper: convert multi-coil k-space to coil images.

    Args:
        kspace: Complex tensor [..., coil, ky, kx].

    Returns:
        images: Complex tensor [..., coil, y, x].
    """
    return ifft2c(kspace)


def image_to_kspace(image: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper: convert coil images back to k-space.

    Args:
        image: Complex tensor [..., coil, y, x].

    Returns:
        kspace: Complex tensor [..., coil, ky, kx].
    """
    return fft2c(image)


# ---------------------------------------------------------------------------
# Sensitivity-map estimation (calibration region based)
# ---------------------------------------------------------------------------

def estimate_sensitivity_maps(
    kspace: torch.Tensor,
    calib_lines: int = 24,
) -> torch.Tensor:
    """
    Estimate coil sensitivity maps from the fully-sampled ACS (Auto-Calibration
    Signal) region at the centre of k-space.

    Method:
        1. Extract the central `calib_lines` rows of k-space.
        2. Zero-pad back to full resolution.
        3. IFFT to image space.
        4. RSS-normalise so sensitivities sum to unit norm at each pixel.

    Args:
        kspace:      Complex tensor [coil, ky, kx] (single frame / slice).
        calib_lines: Number of central k-space lines used for calibration.

    Returns:
        smaps: Complex sensitivity maps of shape [coil, y, x].
    """
    n_coil, ky, kx = kspace.shape

    # Extract central calibration region
    centre = ky // 2
    half   = calib_lines // 2
    calib  = kspace[:, centre - half : centre + half, :]

    # Zero-pad to original k-space size
    pad_ks = torch.zeros_like(kspace)
    pad_ks[:, centre - half : centre + half, :] = calib

    # Transform to image space
    coil_imgs = ifft2c(pad_ks)   # [coil, y, x]

    # Normalise by RSS
    rss = torch.sqrt((coil_imgs.abs() ** 2).sum(dim=0, keepdim=True) + 1e-8)
    smaps = coil_imgs / rss      # [coil, y, x]
    return smaps


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Normalise k-space by its maximum absolute value so the dynamic range
    is in [0, 1].  Returns normalised tensor and the scale factor.

    Args:
        kspace: Complex tensor of any shape.

    Returns:
        (normalised_kspace, scale)  where scale = max(|kspace|).
    """
    scale = kspace.abs().max().item()
    if scale == 0:
        return kspace, 1.0
    return kspace / scale, scale
