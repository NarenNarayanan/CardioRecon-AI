# 🫀 CardioRecon AI

Deep learning reconstruction of undersampled cardiac cine MRI from multi-coil k-space data.

---

## Overview

CardioRecon AI is a modular deep learning system that reconstructs high-quality dynamic cardiac MRI images from undersampled k-space measurements. The system integrates physics-based data consistency with an unrolled U-Net reconstruction network, and ships with a Streamlit web UI for interactive use.

**Dataset:** CMRxRecon (multi-coil short-axis cardiac cine MRI)  

---

## System Pipeline
```
Input: cine_sax_ks.mat  [slice, time, coil, ky, kx]
              ↓
Module 1 — K-space loading & undersampling simulation
              ↓
Module 2 — Hybrid space transformation (centred FFT/IFFT)
              ↓
Module 3 — Unrolled U-Net reconstruction (N cascades)
              ↓
Module 4 — Physics-based data consistency enforcement
              ↓
Module 5 — RSS coil combination → magnitude images
              ↓
Output: reconstructed cine MRI frames + animated GIF
```

---

## Architecture

| Component | Details |
|---|---|
| Model | CardiacMRIReconNet — unrolled reconstruction network |
| Cascades | 5 (configurable) |
| Denoiser | Residual U-Net, 3 levels, 32 base channels |
| Data consistency | Learnable soft DC per cascade |
| Coil combination | Root Sum of Squares (RSS) |
| Parameters | 9,017,550 |
| Loss | L1 + MSE composite |
| Optimiser | AdamW + cosine annealing LR |

---

## Project Structure
```
cardiorecon_ai/
├── data/
│   └── cine_sax_ks.mat          ← CMRxRecon k-space data (not tracked by git)
├── checkpoints/
│   └── best_model.pt            ← trained model weights (not tracked by git)
├── results/                     ← inference outputs (GIFs, PNGs)
├── datasets/
│   └── dataset_loader.py        ← Module 1: k-space loading, sampling masks
├── transforms/
│   └── fft_utils.py             ← Module 2: centred FFT/IFFT (torch.fft)
├── models/
│   ├── unet.py                  ← residual U-Net denoiser
│   └── reconstruction_model.py  ← Module 3: unrolled cascade network
├── reconstruction/
│   └── data_consistency.py      ← Module 4: soft data consistency layer
├── utils/
│   ├── coil_combination.py      ← Module 5: RSS coil combination
│   └── visualization.py         ← GIF export, comparison plots
├── ui/
│   └── app.py                   ← Streamlit web interface
├── train.py                     ← training script
├── inference.py                 ← inference pipeline
├── generate_demo_data.py        ← synthetic k-space generator
└── generate_demo_checkpoint.py  ← demo checkpoint generator
```

---

## Quick Start

### 1. Install dependencies
```bash
# With CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install numpy scipy h5py matplotlib Pillow streamlit imageio tqdm einops
```

### 2. Prepare data

Place your CMRxRecon files in `data/`:
```
data/
├── cine_sax_ks.mat
├── cine_sax_calib.mat
└── cine_sax_info.csv
```

Or generate synthetic data for testing:
```bash
python generate_demo_data.py
```

### 3. Train
```bash
python train.py \
    --ks_path data/cine_sax_ks.mat \
    --n_cascades 5 \
    --epochs 50
# Checkpoint saved to checkpoints/best_model.pt
```

### 4. Run inference
```bash
python inference.py \
    --ks_path    data/cine_sax_ks.mat \
    --ckpt_path  checkpoints/best_model.pt \
    --output_dir results/ \
    --compare \
    --slice_idx  9
```

Outputs:
- `results/cine_recon.gif` — animated reconstructed beating heart
- `results/cine_zerofill.gif` — aliased baseline for comparison
- `results/comparison.png` — side-by-side plot
- `results/frames/` — individual PNG frames

### 5. Launch UI
```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Upload your `.mat` file and `.pt` checkpoint via the sidebar.

---

## Dataset Format

The system processes CMRxRecon-format `.mat` files (HDF5 v7.3):

| File | Contents |
|---|---|
| `cine_sax_ks.mat` | Multi-coil k-space `[slice, time, coil, ky, kx]` |
| `cine_sax_calib.mat` | ACS calibration region |
| `cine_sax_info.csv` | Scan parameters |

Download: [https://cmrxrecon.github.io/](https://cmrxrecon.github.io/)

---

## Training Results

| Setting | Value |
|---|---|
| Epochs | 50 |
| Acceleration | 4× variable density |
| Coils used | 10 of 30 |
| Final loss | 0.00625 |
| GPU | NVIDIA RTX A4000 |
| Time per epoch | ~55 seconds |

---

## Notes on Short-Axis Cardiac MRI

The reconstructed images show a **short-axis cross-section** of the heart — viewed from below, it appears as a circular/oval shape (not the typical heart outline). The left ventricle appears as a **donut shape**: bright myocardium ring surrounding a darker blood pool.

- Slices 0–3: chest wall / base — little visible cardiac structure
- Slices 8–12: mid-ventricle — clearest LV/RV view
- Use `--slice_idx 9` for best results

---

## Reference

## Reference Paper

Wang, Z., et al. "Deep Separable Spatiotemporal Learning for Fast Dynamic Cardiac MRI."
IEEE Transactions on Biomedical Engineering, Vol. 72, No. 12, December 2025.
DOI: 10.1109/TBME.2025.3574090

Our implementation is a simplified version inspired by the DeepSSL architecture,
using a standard unrolled U-Net reconstruction network with physics-based data
consistency, adapted for the CMRxRecon dataset format.

- CMRxRecon Dataset: https://cmrxrecon.github.io/  
                     https://www.synapse.org/Synapse:syn59814210/wiki/628454
- DeepSSL Paper: [https://ieeexplore.ieee.org/document/10960503](https://ieeexplore.ieee.org/document/11016210)

---
