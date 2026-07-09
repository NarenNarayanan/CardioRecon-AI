"""
Module 1 — K-space Acquisition and Undersampling
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Union


def load_kspace(filepath: Union[str, Path], max_coils: int = 10) -> torch.Tensor:
    """
    Load multi-coil k-space. Slices coils ON DISK to save RAM.
    Returns complex64 tensor [time, slice, coil, ky, kx].
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"K-space file not found: {filepath}")

    candidate_keys = ["kspace_data", "kspace", "ks", "data", "kData"]

    with h5py.File(filepath, "r") as f:
        available_keys = list(f.keys())
        print(f"[load_kspace] Available keys: {available_keys}")

        selected_key = None
        for key in candidate_keys:
            if key in f:
                selected_key = key
                break
        if selected_key is None:
            selected_key = available_keys[0]
            print(f"[load_kspace] Using fallback key: '{selected_key}'")
        else:
            print(f"[load_kspace] Using key: '{selected_key}'")

        dataset    = f[selected_key]
        full_shape = dataset.shape
        dtype      = dataset.dtype
        print(f"[load_kspace] Full shape on disk: {full_shape}  dtype: {dtype}")

        ndim = len(full_shape)
        if ndim == 5:
            # CMRxRecon: [S, T, C, ky, kx] — coil is axis 2
            coils_to_load = min(max_coils, full_shape[2])
            print(f"[load_kspace] Loading {coils_to_load}/{full_shape[2]} coils from disk ...")
            raw = dataset[:, :, :coils_to_load, :, :]
        elif ndim == 4:
            coils_to_load = min(max_coils, full_shape[1])
            raw = dataset[:, :coils_to_load, :, :]
        elif ndim == 3:
            coils_to_load = min(max_coils, full_shape[0])
            raw = dataset[:coils_to_load, :, :]
        else:
            raw = dataset[()]

    print(f"[load_kspace] Loaded raw shape: {raw.shape}")
    kspace = _to_complex_tensor(raw)
    del raw
    kspace = _ensure_shape(kspace)
    print(f"[load_kspace] Final shape: {tuple(kspace.shape)} [time, slice, coil, ky, kx]")
    return kspace


def _to_complex_tensor(raw: np.ndarray) -> torch.Tensor:
    # Case 1: CMRxRecon structured dtype ('real', 'imag') named fields
    if raw.dtype.names is not None:
        names = raw.dtype.names
        if 'real' in names and 'imag' in names:
            real = torch.from_numpy(raw['real'].astype(np.float32))
            imag = torch.from_numpy(raw['imag'].astype(np.float32))
            return torch.complex(real, imag)
        if len(names) >= 2:
            real = torch.from_numpy(raw[names[0]].astype(np.float32))
            imag = torch.from_numpy(raw[names[1]].astype(np.float32))
            return torch.complex(real, imag)

    # Case 2: already complex
    if np.iscomplexobj(raw):
        return torch.from_numpy(raw.astype(np.complex64))

    # Case 3: float array with last dim = 2 → (real, imag)
    if raw.ndim >= 1 and raw.shape[-1] == 2:
        return torch.from_numpy(
            (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
        )

    # Case 4: magnitude only
    return torch.from_numpy(raw.astype(np.float32)).to(torch.complex64)


def _ensure_shape(kspace: torch.Tensor) -> torch.Tensor:
    ndim = kspace.ndim
    if ndim == 5:
        # CMRxRecon: [S, T, C, ky, kx] -> [T, S, C, ky, kx]
        kspace = kspace.permute(1, 0, 2, 3, 4).contiguous()
        print(f"[_ensure_shape] Permuted -> {tuple(kspace.shape)}")
        return kspace
    if ndim == 4:
        return kspace.unsqueeze(0)
    if ndim == 3:
        return kspace.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected k-space ndim={ndim}. Expected 3-5.")


def create_sampling_mask(
    ky: int,
    kx: int,
    acceleration: int = 4,
    mask_type: str = "variable_density",
    center_fraction: float = 0.08,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """Returns binary mask [ky, kx]."""
    rng          = np.random.default_rng(seed)
    n_center     = max(1, int(ky * center_fraction))
    center_start = (ky - n_center) // 2
    center_end   = center_start + n_center
    target_lines = max(n_center, ky // acceleration)
    n_outer      = target_lines - n_center

    if mask_type == "cartesian":
        mask_1d = _cartesian_mask_1d(ky, n_outer, center_start, center_end, rng)
    elif mask_type == "random":
        mask_1d = _random_mask_1d(ky, n_outer, center_start, center_end, rng)
    elif mask_type == "variable_density":
        mask_1d = _variable_density_mask_1d(ky, n_outer, center_start, center_end, rng)
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'.")

    mask_2d = torch.from_numpy(mask_1d[:, None]).expand(ky, kx).clone()
    print(f"[create_sampling_mask] {mask_type}  {acceleration}x  "
          f"lines={int(mask_1d.sum())}/{ky}")
    return mask_2d


def _cartesian_mask_1d(ky, n_outer, center_start, center_end, rng):
    mask      = np.zeros(ky, dtype=bool)
    mask[center_start:center_end] = True
    outer_idx = np.concatenate([np.arange(center_start), np.arange(center_end, ky)])
    mask[rng.choice(outer_idx, size=n_outer, replace=False)] = True
    return mask


def _random_mask_1d(ky, n_outer, center_start, center_end, rng):
    mask      = np.zeros(ky, dtype=bool)
    mask[center_start:center_end] = True
    outer_idx = np.concatenate([np.arange(center_start), np.arange(center_end, ky)])
    mask[rng.choice(outer_idx, size=n_outer, replace=False)] = True
    return mask


def _variable_density_mask_1d(ky, n_outer, center_start, center_end, rng):
    mask      = np.zeros(ky, dtype=bool)
    mask[center_start:center_end] = True
    outer_idx = np.concatenate([np.arange(center_start), np.arange(center_end, ky)])
    distances = np.abs(outer_idx - ky / 2.0)
    probs     = 1.0 / (1.0 + 0.1 * distances)
    probs    /= probs.sum()
    mask[rng.choice(outer_idx, size=min(n_outer, len(outer_idx)),
                    replace=False, p=probs)] = True
    return mask


def apply_mask(
    kspace: torch.Tensor,
    mask:   torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (kspace_undersampled, mask_broadcast)."""
    if mask.shape != kspace.shape[-2:]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} != k-space spatial dims "
            f"{tuple(kspace.shape[-2:])}."
        )
    mask_bc      = mask.to(kspace.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    kspace_under = kspace * mask_bc
    print(f"[apply_mask] {mask.float().mean().item()*100:.1f}% of k-space retained.")
    return kspace_under, mask_bc


def load_and_undersample(
    ks_filepath:     Union[str, Path],
    acceleration:    int   = 4,
    mask_type:       str   = "variable_density",
    center_fraction: float = 0.08,
    seed:            int   = 42,
    max_coils:       int   = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kspace_full = load_kspace(ks_filepath, max_coils=max_coils)
    _, _, _, ky, kx = kspace_full.shape
    mask = create_sampling_mask(ky=ky, kx=kx, acceleration=acceleration,
                                mask_type=mask_type, center_fraction=center_fraction,
                                seed=seed)
    kspace_under, _ = apply_mask(kspace_full, mask)
    return kspace_full, kspace_under, mask