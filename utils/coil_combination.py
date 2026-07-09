"""
Utils — Coil Combination
"""

import torch
import numpy as np
from typing import Optional


def root_sum_of_squares(
    coil_images: torch.Tensor,
    coil_dim: int = -3,
) -> torch.Tensor:
    if coil_images.is_complex():
        magnitude = coil_images.abs()
    else:
        magnitude = coil_images.float()
    return torch.sqrt((magnitude ** 2).sum(dim=coil_dim))


def sensitivity_weighted_combination(
    coil_images: torch.Tensor,
    smaps:       torch.Tensor,
    coil_dim:    int = -3,
) -> torch.Tensor:
    numerator   = (smaps.conj() * coil_images).sum(dim=coil_dim)
    denominator = (smaps.abs() ** 2).sum(dim=coil_dim) + 1e-8
    return numerator / denominator


def combine_coils(
    coil_images: torch.Tensor,
    smaps:       Optional[torch.Tensor] = None,
    coil_dim:    int = -3,
) -> torch.Tensor:
    if smaps is not None:
        combined = sensitivity_weighted_combination(coil_images, smaps, coil_dim=coil_dim)
        return combined.abs()
    else:
        return root_sum_of_squares(coil_images, coil_dim=coil_dim)


def normalise_volume(volume: torch.Tensor, percentile: float = 99.0) -> torch.Tensor:
    v_min = volume.min()
    v_max = torch.tensor(np.percentile(volume.float().numpy(), percentile))
    volume = (volume - v_min) / (v_max - v_min + 1e-8)
    volume = volume.clamp(0.0, 1.0)
    return volume


def extract_cine_frames(
    recon_volume: torch.Tensor,
    slice_idx:    int = 0,
) -> np.ndarray:
    if recon_volume.ndim != 4:
        raise ValueError(
            f"Expected 4-D tensor [T, S, H, W], got shape {tuple(recon_volume.shape)}."
        )
    T, S, H, W = recon_volume.shape
    slice_idx = min(slice_idx, S - 1)
    frames_tensor = recon_volume[:, slice_idx, :, :]
    frames_tensor = normalise_volume(frames_tensor)
    return frames_tensor.cpu().numpy().astype(np.float32)
