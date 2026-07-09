"""
Separable Dataset Generator — DeepSSL Paper Implementation
===========================================================
Implements the core innovation of the DeepSSL paper:

"The frequency encoding (FE) direction is always fully sampled.
 By taking the 1D inverse Fourier transform along FE, the 3D k-t
 data can be separated into many independent 2D k-t slices."
                                        — Wang et al., IEEE TBME 2025

This converts ONE 3D subject into M independent 2D training samples,
where M = number of FE points (e.g., 512).

With 100 subjects × 512 FE points = 51,200 training samples
vs direct learning: 100 subjects = 100 training samples

Pipeline:
    3D k-space [M, NJ, T]
          ↓
    k-t Random Undersampling (PE-t space)
          ↓
    1D IFT along FE direction
          ↓
    3D Hybrid data [M, NJ, T]  (image × PE × time)
          ↓
    Extract each FE row → 2D k-t slice [NJ, T]
          ↓
    Real-Imaginary channel splitting → [2, NJ, T]
          ↓
    Ground truth pairing
          ↓
    M independent 2D training samples per subject
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Step 1 — 1D Inverse Fourier Transform along FE direction
# ---------------------------------------------------------------------------

def apply_1d_fe_ifft(kspace_3d: torch.Tensor) -> torch.Tensor:
    """
    Apply 1D Inverse Fourier Transform along the Frequency Encoding (FE)
    direction to convert 3D k-t data into 3D hybrid-space data.

    This is the CORE operation of separable learning.

    Mathematical formulation (Eq. 1 from paper):
        Z = F*_FE × Y = F*_FE × U × F_2D × S × X

    where:
        Y = 3D k-space data  [M × NJ × T]
        F*_FE = 1D IFT along FE (kx) dimension
        Z = 3D hybrid data   [M × NJ × T]
              ↑
        Now each row m is an INDEPENDENT 2D k-t problem

    After this transform:
        - The FE (x) dimension becomes IMAGE space
        - The PE (ky) dimension remains K-SPACE
        - The temporal (t) dimension remains undersampled

    Args:
        kspace_3d: Complex tensor [coil, FE, PE, time] or [FE, coil*PE, time]
                   The FE dimension (kx) is always fully sampled.

    Returns:
        hybrid_3d: Complex tensor of same shape — FE dimension is now
                   in image space, PE dimension remains in k-space.
    """
    # Apply 1D IFFT along the FE (kx) dimension
    # torch.fft.ifft operates along the last dimension by default
    # We apply along dim=1 (FE dimension)

    # Step 1: ifftshift to move DC from centre to corner (torch convention)
    fe_dim = 1
    kspace_shifted = torch.roll(
        kspace_3d,
        shifts=-(kspace_3d.shape[fe_dim] // 2),
        dims=fe_dim
    )

    # Step 2: 1D IFFT along FE direction (orthonormal)
    hybrid = torch.fft.ifft(kspace_shifted, dim=fe_dim, norm="ortho")

    # Step 3: fftshift to move DC back to centre
    hybrid = torch.roll(
        hybrid,
        shifts=hybrid.shape[fe_dim] // 2,
        dims=fe_dim
    )

    print(f"[apply_1d_fe_ifft] 3D k-space → 3D hybrid space")
    print(f"  Input (k-space):  {tuple(kspace_3d.shape)}")
    print(f"  Output (hybrid):  {tuple(hybrid.shape)}")
    print(f"  FE dimension is now in IMAGE space")
    print(f"  PE dimension remains in K-SPACE")
    return hybrid


# ---------------------------------------------------------------------------
# Step 2 — k-t Random Undersampling
# ---------------------------------------------------------------------------

def kt_random_undersampling(
    kspace: torch.Tensor,
    acceleration: int = 6,
    calib_lines: int = 24,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k-t Random Undersampling — different PE lines sampled at each time frame.

    This creates INCOHERENT aliasing across space and time, which is
    essential for the reconstruction algorithms to separate aliased signals.

    Each temporal frame t gets a different random subset of PE lines:
        R_t ⊂ {1, ..., N}  (random PE indices for frame t)
        |R_t| = N / AF      (AF = acceleration factor)

    The undersampling happens ONLY in the PE-t space.
    The FE direction is ALWAYS fully sampled (scanner hardware constraint).

    Args:
        kspace:       Complex tensor [coil, FE, PE, time]
        acceleration: AF = how many times faster (e.g. 6 = keep 1/6 lines)
        calib_lines:  Number of fully-sampled centre lines (ACS region)
                      These are needed for coil sensitivity estimation
        seed:         Random seed for reproducibility

    Returns:
        kspace_under: Undersampled k-space (zero-filled at unsampled positions)
        mask:         Binary mask [PE, time]  (1=sampled, 0=unsampled)
    """
    _, _, n_pe, n_time = kspace.shape
    rng = np.random.default_rng(seed)

    mask = np.zeros((n_pe, n_time), dtype=np.float32)

    # Always fully sample the central ACS (Auto-Calibration Signal) region
    centre = n_pe // 2
    half_calib = calib_lines // 2
    mask[centre - half_calib : centre + half_calib, :] = 1.0

    # For each time frame, randomly sample outer PE lines
    lines_per_frame = n_pe // acceleration
    outer_lines = list(range(0, centre - half_calib)) + \
                  list(range(centre + half_calib, n_pe))

    for t in range(n_time):
        # Different random subset for each frame → incoherent aliasing
        sampled = rng.choice(outer_lines, size=lines_per_frame, replace=False)
        mask[sampled, t] = 1.0

    mask_tensor = torch.from_numpy(mask)  # [PE, time]

    # Apply mask: broadcast across coil and FE dimensions
    # mask [PE, time] → [1, 1, PE, time]
    mask_bc = mask_tensor.unsqueeze(0).unsqueeze(0)
    kspace_under = kspace * mask_bc

    sampled_pct = mask.mean() * 100
    print(f"[kt_random_undersampling] AF={acceleration}x  "
          f"sampled={sampled_pct:.1f}%  calib_lines={calib_lines}")
    return kspace_under, mask_tensor


# ---------------------------------------------------------------------------
# Step 3 — Extract 2D k-t Slices (the separable decomposition)
# ---------------------------------------------------------------------------

def extract_2d_kt_slices(
    hybrid_3d: torch.Tensor,
    fullsamp_3d: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract independent 2D k-t slices from 3D hybrid data.

    After the 1D FE IFFT, each row along the FE dimension is an
    INDEPENDENT 2D k-t reconstruction problem.

    From the paper (Section III-A):
        "Given the 1D IFT decoupling, the m-th row data Zm ∈ C^{NJ×T}
         of Z can be treated as an independent 2D k-t signal."

    This is the data augmentation step:
        1 subject × M FE points → M independent training samples

    Args:
        hybrid_3d:    Undersampled 3D hybrid data [coil, FE=M, PE, time]
        fullsamp_3d:  Fully-sampled 3D image data  [FE=M, PE, time]

    Returns:
        List of (input_2d, label_2d) pairs:
            input_2d: undersampled 2D k-t  [2*coil, PE, time]  (re+im channels)
            label_2d: fully-sampled 2D image [PE, time]         (magnitude)
    """
    n_coil, n_fe, n_pe, n_time = hybrid_3d.shape
    pairs = []

    for m in range(n_fe):
        # Extract m-th row from undersampled hybrid data
        # Shape: [coil, PE, time] complex
        slice_under = hybrid_3d[:, m, :, :]

        # Extract m-th row from fully-sampled image (ground truth label)
        # Shape: [PE, time] real (magnitude)
        slice_full = fullsamp_3d[m, :, :]

        # Split complex into real + imaginary channels
        # [coil, PE, time] complex → [2*coil, PE, time] real
        slice_real = torch.cat([slice_under.real, slice_under.imag], dim=0)

        pairs.append((slice_real, slice_full.abs()))

    print(f"[extract_2d_kt_slices] Extracted {len(pairs)} independent 2D samples")
    print(f"  Each input  shape: {pairs[0][0].shape}  (2*coil, PE, time)")
    print(f"  Each label  shape: {pairs[0][1].shape}  (PE, time)")
    print(f"  Data augmentation: 1 subject × {n_fe} FE rows = {n_fe} samples")
    return pairs


# ---------------------------------------------------------------------------
# PyTorch Dataset — Wraps the full pipeline
# ---------------------------------------------------------------------------

class SeparableKtDataset(Dataset):
    """
    PyTorch Dataset implementing DeepSSL separable learning.

    For each subject:
        1. Load 3D k-space
        2. Apply k-t random undersampling
        3. Apply 1D FE IFFT → hybrid domain
        4. Extract M independent 2D k-t slices
        5. Each slice becomes one training sample

    Result: N_subjects × M_FE total training samples
            (vs N_subjects for direct learning)

    From Table I of the paper:
        Direct learning:    N_TC × N_slice training samples
        Separable learning: N_TC × N_slice × M training samples
    """

    def __init__(
        self,
        ks_files:     List[str],
        acceleration: int = 6,
        calib_lines:  int = 24,
    ):
        self.samples = []

        for ks_file in ks_files:
            print(f"[SeparableKtDataset] Processing: {ks_file}")
            pairs = self._process_subject(ks_file, acceleration, calib_lines)
            self.samples.extend(pairs)

        print(f"[SeparableKtDataset] Total samples: {len(self.samples)}")

    def _process_subject(self, ks_file, acceleration, calib_lines):
        with h5py.File(ks_file, "r") as f:
            raw = f[list(f.keys())[0]][()]
            real = torch.from_numpy(raw['real'].astype(np.float32))
            imag = torch.from_numpy(raw['imag'].astype(np.float32))
            kspace = torch.complex(real, imag)

        # kspace shape: [slice, time, coil, PE, FE]
        # Rearrange to: [coil, FE, PE, time] for one slice
        kspace = kspace.permute(2, 4, 3, 1)  # → [coil, FE, PE, time]

        # Step 1: Apply k-t undersampling
        kspace_under, mask = kt_random_undersampling(
            kspace, acceleration, calib_lines
        )

        # Step 2: Apply 1D FE IFFT → hybrid domain
        hybrid_under = apply_1d_fe_ifft(kspace_under)
        hybrid_full  = apply_1d_fe_ifft(kspace)         # ground truth

        # Step 3: Extract 2D k-t slices → M training samples
        pairs = extract_2d_kt_slices(hybrid_under, hybrid_full)
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_2d, label_2d = self.samples[idx]
        return input_2d, label_2d