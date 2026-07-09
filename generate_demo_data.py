"""
generate_demo_data.py
======================
Generates SYNTHETIC cardiac MRI k-space data for testing the full pipeline
WITHOUT needing the real CMRxRecon dataset.

The synthetic data is physically realistic:
  - Multi-coil k-space generated from a simulated cardiac phantom
  - Dynamic frames (beating heart simulation via radius modulation)
  - Correct HDF5/.mat format matching CMRxRecon conventions
  - Realistic k-space energy distribution

Output:
    data/cine_sax_ks.mat      ← synthetic k-space  [time, slice, coil, ky, kx]
    data/cine_sax_calib.mat   ← calibration region [coil, ky_calib, kx]
    data/cine_sax_info.csv    ← dummy scan metadata

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --n_frames 12 --n_slices 3 --n_coils 8 --size 128
"""

import argparse
import numpy as np
import h5py
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Cardiac phantom
# ---------------------------------------------------------------------------

def make_cardiac_phantom(
    size:     int,
    frame_idx: int,
    n_frames:  int,
) -> np.ndarray:
    """
    Simulate a 2-D short-axis cardiac MRI slice for one time frame.

    Anatomy modelled:
        • Left ventricle  (LV)  — hollow, with wall
        • Right ventricle (RV)  — crescent shape
        • Myocardium             — bright ring around LV
        • Background             — dark chest tissue

    The LV radius oscillates sinusoidally to simulate the beating heart.

    Returns:
        image: float32 array [size, size], values in [0, 1]
    """
    H, W = size, size
    cx, cy = W // 2, H // 2

    # Cardiac phase: 0 = end-diastole (largest), 1 = end-systole (smallest)
    phase = (1.0 - np.cos(2 * np.pi * frame_idx / n_frames)) / 2.0

    # ── Coordinate grid ────────────────────────────────────────────────────
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy -= cy
    xx -= cx

    # ── Left ventricle (LV) ────────────────────────────────────────────────
    lv_r_outer = size * (0.20 + 0.04 * (1.0 - phase))   # diastole = bigger
    lv_r_inner = size * (0.10 + 0.05 * phase)             # systole  = thicker wall

    dist_lv = np.sqrt(xx**2 + yy**2)
    lv_wall  = (dist_lv <= lv_r_outer) & (dist_lv >  lv_r_inner)   # myocardium
    lv_blood = (dist_lv <= lv_r_inner)                               # blood pool

    # ── Right ventricle (RV) ───────────────────────────────────────────────
    # Offset to the right
    rv_cx = size * 0.18
    rv_r  = size * (0.14 + 0.02 * (1.0 - phase))
    dist_rv = np.sqrt((xx - rv_cx)**2 + yy**2)
    rv_mask = (dist_rv <= rv_r) & (dist_lv > lv_r_outer * 0.9)     # crescent

    # ── Papillary muscles (small dark spots inside LV) ────────────────────
    pm1 = np.sqrt((xx - lv_r_inner * 0.5)**2 + (yy + lv_r_inner * 0.3)**2)
    pm2 = np.sqrt((xx + lv_r_inner * 0.5)**2 + (yy + lv_r_inner * 0.3)**2)
    pap = (pm1 < size * 0.025) | (pm2 < size * 0.025)

    # ── Assemble image ────────────────────────────────────────────────────
    image = np.zeros((H, W), dtype=np.float32)
    image[lv_wall]  = 0.85    # myocardium — bright
    image[lv_blood] = 0.60    # blood pool — slightly less bright
    image[rv_mask]  = 0.50    # RV blood
    image[pap & lv_blood] = 0.20   # papillary muscles — dark

    # Add mild background body tissue
    body = np.sqrt(xx**2 + (yy * 0.7)**2)
    image[body < size * 0.45] = np.maximum(
        image[body < size * 0.45], 0.08
    )

    # Mild Gaussian blur to simulate partial volume effect
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=size * 0.012)

    # Add subtle Rician noise
    rng = np.random.default_rng(seed=frame_idx * 1000)
    noise = rng.normal(0, 0.015, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)

    return image


# ---------------------------------------------------------------------------
# Coil sensitivity maps
# ---------------------------------------------------------------------------

def make_sensitivity_maps(
    size:    int,
    n_coils: int,
) -> np.ndarray:
    """
    Generate synthetic coil sensitivity maps.

    Coils are arranged in a ring around the object.
    Each sensitivity map peaks near the coil element and falls off smoothly.

    Returns:
        smaps: complex64 array [n_coils, size, size]
    """
    H, W = size, size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy = (yy - H / 2) / (H / 2)
    xx = (xx - W / 2) / (W / 2)

    smaps = np.zeros((n_coils, H, W), dtype=np.complex64)

    for c in range(n_coils):
        # Place coil element on a circle of radius 1.5 (outside the FOV)
        angle = 2 * np.pi * c / n_coils
        coil_x = 1.5 * np.cos(angle)
        coil_y = 1.5 * np.sin(angle)

        # Distance from coil element
        dist = np.sqrt((xx - coil_x)**2 + (yy - coil_y)**2)

        # Magnitude: Gaussian fall-off from coil position
        magnitude = np.exp(-dist**2 / (2 * 0.6**2)).astype(np.float32)

        # Phase: linear gradient pointing toward coil (simple model)
        phase = (coil_x * xx + coil_y * yy) * 0.8

        smaps[c] = magnitude * np.exp(1j * phase)

    # Normalise: unit RSS at each pixel
    rss = np.sqrt((np.abs(smaps)**2).sum(axis=0, keepdims=True)) + 1e-8
    smaps /= rss

    return smaps


# ---------------------------------------------------------------------------
# Forward model: image → multi-coil k-space
# ---------------------------------------------------------------------------

def image_to_multicoil_kspace(
    image: np.ndarray,       # [H, W]  float
    smaps: np.ndarray,       # [C, H, W] complex
) -> np.ndarray:
    """
    Simulate multi-coil k-space acquisition.

        coil_image_c = image * sensitivity_c
        kspace_c     = FFT(coil_image_c)

    Returns:
        kspace: complex64 [n_coils, ky, kx]
    """
    n_coils = smaps.shape[0]
    H, W    = image.shape
    kspace  = np.zeros((n_coils, H, W), dtype=np.complex64)

    for c in range(n_coils):
        coil_img = image.astype(np.complex64) * smaps[c]
        # Centred FFT
        kspace[c] = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(coil_img))
        ) / np.sqrt(H * W)

    return kspace


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_demo_data(
    output_dir: str  = "data",
    n_frames:   int  = 12,
    n_slices:   int  = 3,
    n_coils:    int  = 10,
    size:       int  = 128,
    verbose:    bool = True,
):
    """
    Generate synthetic CMRxRecon-format k-space data.

    Creates:
        data/cine_sax_ks.mat     — k-space tensor [T, S, C, ky, kx] (complex as re/im)
        data/cine_sax_calib.mat  — calibration lines [C, ky_calib, kx]
        data/cine_sax_info.csv   — scan metadata

    Args:
        output_dir: Folder to write files (created if absent).
        n_frames:   Number of cardiac phases (temporal frames).
        n_slices:   Number of short-axis slices.
        n_coils:    Number of receive coils.
        size:       Image size (size × size pixels).
        verbose:    Print progress messages.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(f"[generate_demo_data] {msg}")

    log(f"Generating synthetic k-space: "
        f"T={n_frames}  S={n_slices}  C={n_coils}  size={size}×{size}")

    # ── Sensitivity maps (same for all slices/frames) ──────────────────────
    log("Building coil sensitivity maps …")
    smaps = make_sensitivity_maps(size, n_coils)    # [C, H, W]

    # ── K-space tensor [T, S, C, ky, kx] ──────────────────────────────────
    log("Simulating dynamic cardiac phantom + forward model …")
    kspace = np.zeros((n_frames, n_slices, n_coils, size, size),
                      dtype=np.complex64)

    for si in range(n_slices):
        # Slightly shift slice centre to mimic different anatomical levels
        slice_offset = (si - n_slices // 2) * 0.05
        for fi in range(n_frames):
            img = make_cardiac_phantom(size, fi, n_frames)
            # Shift phantom slightly per slice
            img = np.roll(img, int(slice_offset * size), axis=0)

            ks = image_to_multicoil_kspace(img, smaps)    # [C, ky, kx]
            kspace[fi, si] = ks

        log(f"  Slice {si+1}/{n_slices} done")

    # Normalise to unit max magnitude
    kspace /= (np.abs(kspace).max() + 1e-8)

    # ── Save cine_sax_ks.mat ───────────────────────────────────────────────
    ks_path = out / "cine_sax_ks.mat"
    log(f"Saving k-space → {ks_path}  (shape {kspace.shape})")
    with h5py.File(ks_path, "w") as f:
        # Store as real/imaginary float32 arrays (common CMRxRecon format)
        f.create_dataset("kspace_data_real", data=kspace.real, compression="gzip")
        f.create_dataset("kspace_data_imag", data=kspace.imag, compression="gzip")
        # Also store complex directly (h5py supports this)
        f.create_dataset("kspace_data",      data=kspace,      compression="gzip")
        f.attrs["shape"]       = list(kspace.shape)
        f.attrs["description"] = "Synthetic cardiac cine k-space [T, S, C, ky, kx]"
        f.attrs["n_frames"]    = n_frames
        f.attrs["n_slices"]    = n_slices
        f.attrs["n_coils"]     = n_coils
        f.attrs["image_size"]  = size

    # ── Save cine_sax_calib.mat ────────────────────────────────────────────
    # Extract central 24 lines as calibration region
    calib_lines = 24
    cy = size // 2
    calib = kspace[0, 0, :, cy-calib_lines//2 : cy+calib_lines//2, :]  # [C, 24, kx]

    calib_path = out / "cine_sax_calib.mat"
    log(f"Saving calibration data → {calib_path}  (shape {calib.shape})")
    with h5py.File(calib_path, "w") as f:
        f.create_dataset("calib_data",      data=calib,       compression="gzip")
        f.create_dataset("calib_data_real", data=calib.real,  compression="gzip")
        f.create_dataset("calib_data_imag", data=calib.imag,  compression="gzip")
        f.attrs["description"] = "ACS calibration region [C, ky_calib, kx]"
        f.attrs["n_coils"]     = n_coils
        f.attrs["calib_lines"] = calib_lines

    # ── Save cine_sax_info.csv ─────────────────────────────────────────────
    info_path = out / "cine_sax_info.csv"
    log(f"Saving scan info → {info_path}")
    with open(info_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Parameter", "Value", "Unit"])
        writer.writerow(["n_frames",       n_frames,    "frames"])
        writer.writerow(["n_slices",       n_slices,    "slices"])
        writer.writerow(["n_coils",        n_coils,     "coils"])
        writer.writerow(["image_size",     f"{size}x{size}", "pixels"])
        writer.writerow(["FOV",            "360x360",   "mm"])
        writer.writerow(["resolution",     f"{360/size:.1f}x{360/size:.1f}", "mm/pixel"])
        writer.writerow(["TR",             "35.5",      "ms"])
        writer.writerow(["TE",             "1.5",       "ms"])
        writer.writerow(["flip_angle",     "50",        "degrees"])
        writer.writerow(["heart_rate",     "65",        "bpm"])
        writer.writerow(["dataset_type",   "SYNTHETIC", ""])
        writer.writerow(["generated_by",   "generate_demo_data.py", ""])

    log("")
    log("✅  Synthetic dataset ready!")
    log(f"   K-space file:    {ks_path}  ({_file_mb(ks_path):.1f} MB)")
    log(f"   Calibration:     {calib_path}  ({_file_mb(calib_path):.1f} MB)")
    log(f"   Scan info:       {info_path}")
    log("")
    log("Next steps:")
    log("  1. Generate a demo checkpoint:  python generate_demo_checkpoint.py")
    log("  2. Run inference:               python inference.py --ks_path data/cine_sax_ks.mat --ckpt_path checkpoints/demo_model.pt")
    log("  3. Launch the UI:               streamlit run ui/app.py")


def _file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate synthetic cardiac MRI k-space data for testing."
    )
    p.add_argument("--output_dir", type=str, default="data",
                   help="Output directory (default: data/)")
    p.add_argument("--n_frames",   type=int, default=12,
                   help="Number of cardiac phases / temporal frames (default 12)")
    p.add_argument("--n_slices",   type=int, default=3,
                   help="Number of short-axis slices (default 3)")
    p.add_argument("--n_coils",    type=int, default=10,
                   help="Number of receive coils (default 10)")
    p.add_argument("--size",       type=int, default=128,
                   help="Image size in pixels (default 128 → 128×128)")
    args = p.parse_args()

    generate_demo_data(
        output_dir = args.output_dir,
        n_frames   = args.n_frames,
        n_slices   = args.n_slices,
        n_coils    = args.n_coils,
        size       = args.size,
    )
