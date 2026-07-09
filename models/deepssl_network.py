"""
DeepSSL Network — Deep Separable Spatiotemporal Learning
=========================================================
Implements the full DeepSSL architecture from:

    Wang et al., "Deep Separable Spatiotemporal Learning for
    Fast Dynamic Cardiac MRI", IEEE TBME, Vol.72, No.12, 2025.
    DOI: 10.1109/TBME.2025.3574090

Architecture (Fig. 4 from paper):

    Input: undersampled 2D k-t slice  [B, 2C, PE, T]
              ↓
    K network phases (K=10):
      ┌─────────────────────────────────────┐
      │  Deep Temporal Low-Rank Module      │  ← 1D conv on temporal signals
      │  Deep Spatial Sparse Module         │  ← learned soft-thresholding
      │  Data Consistency Module            │  ← weighted k-space fusion
      └─────────────────────────────────────┘
              ↓
    Output: reconstructed 2D spatiotemporal image [B, PE, T]

Key innovation: ALL convolutions are 1D — either temporal (along T)
or spatial (along PE). Never 2D. This is what makes separable learning work.

Unrolled from the optimisation problem (Eq. 2 from paper):
    min_{Xm} (1/2)||Zm - A×Xm||²
           + λ₁ Σₙ ||P(Q)·En·Xm||²_F     ← temporal low-rank term
           + λ₂ Σₜ ||D·Vt·Xm||₁           ← spatial sparsity term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Module 1 — Deep Temporal Low-Rank Module
# ---------------------------------------------------------------------------

class DeepTemporalLowRankModule(nn.Module):
    """
    Deep Temporal Low-Rank Module (DTLR)

    Corresponds to sub-problem 1 of the optimisation (Eq. 3 & 7):

        b^(k)_{m,n} = En·X^(k-1)_m - N1(En·X^(k-1)_m)

    Intuition:
        - Extract each temporal signal EnXm (signal at position n across all T frames)
        - Apply learnable null space projection N1 to estimate artifacts
        - Subtract artifacts → residual learning for temporal de-aliasing

    Why this enforces low-rank:
        Fully sampled cardiac MRI has low-rank temporal structure (heart beats
        periodically → temporal signals lie on a low-dimensional manifold).
        Undersampling introduces high-rank artifacts. N1 learns to identify
        and remove these artifacts, pushing the output toward low-rank structure.

    From paper (Section III-D-1):
        "We directly use the multi-layer convolutional network to replace
         (P(Q))^H · P(Q) to achieve nonlinear null space projection"

    Network N1:
        - 6 convolutional layers
        - 48 filters of size 3
        - 1D convolution along TEMPORAL dimension only
        - ReLU activations
        - Last layer: 2 filters (real and imaginary channels)
    """

    def __init__(
        self,
        n_channels:  int = 2,     # 2 = real + imaginary
        n_filters:   int = 48,    # 48 filters as specified in paper
        filter_size: int = 3,     # 1D filter size along temporal dim
        n_layers:    int = 6,     # 6 conv layers as in paper
    ):
        super().__init__()

        # Build N1: multi-layer 1D CNN operating along TEMPORAL dimension
        # Input: [B, n_channels, T]  (temporal signals)
        # Output: [B, n_channels, T] (estimated artifact/null space component)
        layers = []

        # First layer: n_channels → n_filters
        layers.append(
            nn.Conv1d(n_channels, n_filters,
                      kernel_size=filter_size, padding=filter_size//2)
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: n_filters → n_filters
        for _ in range(n_layers - 2):
            layers.append(
                nn.Conv1d(n_filters, n_filters,
                          kernel_size=filter_size, padding=filter_size//2)
            )
            layers.append(nn.ReLU(inplace=True))

        # Last layer: n_filters → n_channels (real + imaginary)
        # No activation on last layer
        layers.append(
            nn.Conv1d(n_filters, n_channels,
                      kernel_size=filter_size, padding=filter_size//2)
        )

        self.N1 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal low-rank module.

        Args:
            x: 2D k-t slice [B, 2C, PE, T]
               (2C channels = real and imaginary parts of C coils)

        Returns:
            b: Temporally de-aliased estimate [B, 2C, PE, T]

        Process:
            For each spatial position n (PE direction):
                1. Extract temporal signal: [B, 2C, T]
                2. Apply N1 → estimated null space (artifacts)
                3. Subtract → de-aliased temporal signal
        """
        B, C2, PE, T = x.shape

        # Process each PE position independently (temporal signals)
        # Reshape: [B, 2C, PE, T] → [B*PE, 2C, T]
        # This applies N1 to EVERY temporal signal En·Xm simultaneously
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * PE, C2, T)

        # N1: estimate the null space component (artifacts)
        # This implements: N1(En·X^(k-1)_m)
        null_space_estimate = self.N1(x_reshaped)

        # Residual subtraction: remove artifacts
        # b^(k)_{m,n} = En·X^(k-1)_m - N1(En·X^(k-1)_m)
        b_reshaped = x_reshaped - null_space_estimate

        # Reshape back: [B*PE, 2C, T] → [B, 2C, PE, T]
        b = b_reshaped.reshape(B, PE, C2, T).permute(0, 2, 1, 3)
        return b


# ---------------------------------------------------------------------------
# Module 2 — Deep Spatial Sparse Module
# ---------------------------------------------------------------------------

class DeepSpatialSparseModule(nn.Module):
    """
    Deep Spatial Sparse Module (DSS)

    Corresponds to sub-problem 2 of the optimisation (Eq. 4 & 8):

        d^(k)_{m,t} = N3[ soft( N2(Vt·X^(k-1)_m) ; θ^(k) ) ]

    Intuition:
        - Extract each spatial signal VtXm (spatial image at time t)
        - Apply learned transform N2 to analysis sparse domain
        - Apply soft-thresholding to shrink small coefficients (remove noise)
        - Apply learned transform N3 to synthesise back to image domain
        - Result: spatially de-aliased image

    From paper (Section III-D-2):
        "We utilise the widely used deep thresholding network to learn
         a more general sparse transform from training datasets."

    Network structure:
        N2: 3 conv layers (analysis transform — learns sparse representation)
        soft-thresholding: element-wise shrinkage with learnable threshold θ
        N3: 3 conv layers (synthesis transform — reconstructs from sparse domain)
        N2 and N3 have NO invertibility constraint (flexible)
    """

    def __init__(
        self,
        n_channels:  int = 2,
        n_filters:   int = 48,
        filter_size: int = 3,
        n_layers:    int = 3,     # 3 layers each for N2 and N3
    ):
        super().__init__()

        # N2: Analysis transform (image → sparse domain)
        # 1D convolution along SPATIAL (PE) dimension
        self.N2 = self._build_1d_cnn(n_channels, n_filters, filter_size,
                                       n_layers, last_channels=n_filters)

        # Learnable threshold θ^(k) — initialised to 0.001 as in paper
        self.theta = nn.Parameter(torch.tensor(0.001))

        # N3: Synthesis transform (sparse domain → image)
        self.N3 = self._build_1d_cnn(n_filters, n_filters, filter_size,
                                       n_layers, last_channels=n_channels)

    def _build_1d_cnn(self, in_ch, hidden_ch, k_size, n_layers, last_channels):
        """Build a 1D CNN with n_layers layers."""
        layers = []
        layers.append(nn.Conv1d(in_ch, hidden_ch,
                                kernel_size=k_size, padding=k_size//2))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(n_layers - 2):
            layers.append(nn.Conv1d(hidden_ch, hidden_ch,
                                    kernel_size=k_size, padding=k_size//2))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(hidden_ch, last_channels,
                                kernel_size=k_size, padding=k_size//2))
        return nn.Sequential(*layers)

    def soft_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Element-wise soft-thresholding (Eq. 4 from paper):

            soft(x; θ) = sign(x) × max(|x| - θ, 0)

        For complex-valued data via real/imaginary channels:
            - Shrinks coefficient magnitudes toward zero
            - Zeros out small coefficients (noise/incoherent aliasing)
            - Preserves large coefficients (true signal)

        θ is LEARNABLE — network decides how aggressively to threshold.
        """
        return torch.sign(x) * F.relu(x.abs() - self.theta.abs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial sparse module.

        Args:
            x: 2D k-t slice [B, 2C, PE, T]

        Returns:
            d: Spatially de-aliased estimate [B, 2C, PE, T]

        Process:
            For each temporal frame t (time direction):
                1. Extract spatial signal: [B, 2C, PE]
                2. Apply N2 → sparse domain coefficients
                3. Soft-threshold → remove incoherent aliasing
                4. Apply N3 → back to image domain
        """
        B, C2, PE, T = x.shape

        # Process each time frame independently (spatial signals)
        # Reshape: [B, 2C, PE, T] → [B*T, 2C, PE]
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B * T, C2, PE)

        # N2: Analysis transform → sparse domain
        # Implements: N2(Vt·X^(k-1)_m)
        sparse_coeffs = self.N2(x_reshaped)

        # Soft-thresholding with learnable θ
        # Implements: soft(N2(Vt·X^(k-1)_m); θ^(k))
        thresholded = self.soft_threshold(sparse_coeffs)

        # N3: Synthesis transform → back to image domain
        # Implements: N3[soft(N2(Vt·X^(k-1)_m); θ^(k))]
        d_reshaped = self.N3(thresholded)

        # Reshape back: [B*T, 2C, PE] → [B, 2C, PE, T]
        d = d_reshaped.reshape(B, T, C2, PE).permute(0, 2, 3, 1)
        return d


# ---------------------------------------------------------------------------
# Module 3 — Data Consistency Module
# ---------------------------------------------------------------------------

class WeightedDataConsistencyModule(nn.Module):
    """
    Weighted Data Consistency Module

    Corresponds to sub-problem 3 of the optimisation (Eq. 5 & 9):

    At MEASURED positions p ∈ Ω:
        k_out[p] = (Zm[p] + μ₁·F(Bm)[p] + μ₂·F(Dm)[p]) / (1 + μ₁ + μ₂)

    At UNMEASURED positions p ∉ Ω:
        k_out[p] = (μ₁·F(Bm)[p] + μ₂·F(Dm)[p]) / (μ₁ + μ₂)

    Where:
        Zm   = measured undersampled k-space
        Bm   = output of temporal low-rank module
        Dm   = output of spatial sparse module
        μ₁   = learnable weight for temporal contribution (learned: 1.98)
        μ₂   = learnable weight for spatial contribution  (learned: 0.23)
        F    = Fourier transform (PE direction only — hybrid domain)

    Key insight from learned μ values (from paper):
        μ₁=1.98 >> μ₂=0.23
        → Temporal low-rank processing is ~8.6× more important than spatial
        → Heart beating = strong temporal structure = most valuable prior
    """

    def __init__(self):
        super().__init__()

        # Learnable trade-off parameters — initialised to 1.0 as in paper
        # Paper reports final learned values: μ₁=1.98, μ₂=0.23
        self.mu1 = nn.Parameter(torch.tensor(1.0))  # temporal weight
        self.mu2 = nn.Parameter(torch.tensor(1.0))  # spatial weight

    def pe_fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D Fourier Transform along PE dimension only.
        Converts from hybrid domain (image×PE) to full k-space.
        """
        return torch.fft.fft(x, dim=2, norm="ortho")

    def pe_inverse_fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D Inverse Fourier Transform along PE dimension only.
        """
        return torch.fft.ifft(x, dim=2, norm="ortho")

    def forward(
        self,
        x_current: torch.Tensor,   # current estimate [B, 2C, PE, T]
        b_temporal: torch.Tensor,   # temporal low-rank output [B, 2C, PE, T]
        d_spatial: torch.Tensor,    # spatial sparse output [B, 2C, PE, T]
        kspace_measured: torch.Tensor,  # measured k-space [B, 2C, PE, T]
        mask: torch.Tensor,         # sampling mask [PE, T]
    ) -> torch.Tensor:
        """
        Apply weighted data consistency.

        The two module outputs (temporal and spatial) are blended
        with the measured data using learnable weights μ₁ and μ₂.
        """
        mu1 = self.mu1.abs()   # ensure positive
        mu2 = self.mu2.abs()

        # Transform B and D outputs to k-space (PE Fourier transform)
        B_kspace = self.pe_fourier_transform(b_temporal)
        D_kspace = self.pe_fourier_transform(d_spatial)
        Z_m      = self.pe_fourier_transform(kspace_measured)

        # Broadcast mask to [1, 1, PE, T]
        mask_bc = mask.unsqueeze(0).unsqueeze(0).to(x_current.device)

        # At measured positions: blend with real measurements
        # k_out[measured] = (Z + μ₁B + μ₂D) / (1 + μ₁ + μ₂)
        numerator_measured = Z_m + mu1 * B_kspace + mu2 * D_kspace
        denominator_measured = 1.0 + mu1 + mu2

        # At unmeasured positions: use weighted combination of modules only
        # k_out[unmeasured] = (μ₁B + μ₂D) / (μ₁ + μ₂)
        numerator_unmeasured = mu1 * B_kspace + mu2 * D_kspace
        denominator_unmeasured = mu1 + mu2

        k_out = (mask_bc * numerator_measured / denominator_measured
                 + (1 - mask_bc) * numerator_unmeasured / denominator_unmeasured)

        # Transform back to hybrid domain
        x_out = self.pe_inverse_fourier_transform(k_out)
        return x_out.real   # return real part of hybrid domain data


# ---------------------------------------------------------------------------
# Full DeepSSL Network — Unrolled K=10 phases
# ---------------------------------------------------------------------------

class DeepSSL(nn.Module):
    """
    Deep Separable Spatiotemporal Learning Network (DeepSSL)

    Full unrolled network with K=10 phases.
    Each phase = DTLR + DSS + DC (three modules from paper).

    From paper implementation details (Section III-E):
        K = 10 network phases
        N1: 6 conv layers, 48 filters, size 3
        N2: 3 conv layers, 48 filters, size 3
        N3: 3 conv layers, 48 filters, size 3
        Batch size: 64
        Optimizer: Adam, lr=0.001, decay=0.99
        Training: 50 epochs, ~50 hours on Tesla T4

    Input shape:  [B, 2C, PE, T]   (2C = real + imag channels for C coils)
    Output shape: [B, PE, T]        (reconstructed 2D spatiotemporal image)
    """

    def __init__(
        self,
        n_channels: int = 2,    # 2 = real + imag (single coil demo)
        n_filters:  int = 48,   # 48 as specified in paper
        K:          int = 10,   # 10 unrolled phases as in paper
    ):
        super().__init__()
        self.K = K

        # Each phase has its OWN set of three modules
        # (NOT shared weights — each phase learns different corrections)
        self.temporal_modules = nn.ModuleList([
            DeepTemporalLowRankModule(n_channels, n_filters)
            for _ in range(K)
        ])
        self.spatial_modules = nn.ModuleList([
            DeepSpatialSparseModule(n_channels, n_filters)
            for _ in range(K)
        ])
        self.dc_modules = nn.ModuleList([
            WeightedDataConsistencyModule()
            for _ in range(K)
        ])

    def forward(
        self,
        kspace_under: torch.Tensor,  # [B, 2C, PE, T]
        mask: torch.Tensor,          # [PE, T]
    ) -> torch.Tensor:
        """
        Run K unrolled reconstruction phases.

        Each phase refines the estimate:
            Phase 1: Coarse artifact removal
            Phase 5: Medium refinement
            Phase 10: Final high-quality reconstruction

        The loss is computed at EVERY phase (not just final) to
        provide supervision throughout the unrolling.
        """
        # Initialise with zero-filled input (direct IFFT of undersampled k-space)
        # X^(0) = A* × Zm  (adjoint encoding operator applied to measurements)
        x = kspace_under

        # Store intermediate outputs for multi-phase loss
        intermediate_outputs = []

        for k in range(self.K):
            # Sub-problem 1: Temporal Low-Rank (Eq. 7)
            # b^(k) = En·X^(k-1) - N1(En·X^(k-1))
            b = self.temporal_modules[k](x)

            # Sub-problem 2: Spatial Sparse (Eq. 8)
            # d^(k) = N3[soft(N2(Vt·X^(k-1)); θ^(k))]
            d = self.spatial_modules[k](x)

            # Sub-problem 3: Data Consistency (Eq. 9)
            # X^(k) = weighted blend of b, d, and measurements
            x = self.dc_modules[k](x, b, d, kspace_under, mask)

            # Store for multi-phase supervision during training
            intermediate_outputs.append(x)

        # Return all phases (for multi-phase loss) and final output
        return intermediate_outputs


# ---------------------------------------------------------------------------
# Multi-Phase Loss Function (Eq. 10 from paper)
# ---------------------------------------------------------------------------

class DeepSSLLoss(nn.Module):
    """
    Multi-Phase L2 Loss Function from paper (Eq. 10):

        L(Θ) = (1/KC) Σ_{k=1}^{K} Σ_{c=1}^{C} ||X^{ref,c}_m - X^{(k),c}_m||²₂

    Key difference from standard deep learning:
        Loss computed at EVERY phase k, not just final output.
        This forces each intermediate reconstruction to also be good.
        Result: more stable training, better convergence.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        phase_outputs: list,   # list of K tensors, each [B, 2C, PE, T]
        target: torch.Tensor,  # [B, PE, T] ground truth
    ) -> torch.Tensor:
        """Compute sum of L2 losses across all K phases."""
        K = len(phase_outputs)
        total_loss = torch.tensor(0.0, requires_grad=True)

        for k, x_k in enumerate(phase_outputs):
            # Take magnitude of output at phase k
            # Stack real+imag → magnitude: sqrt(re² + im²)
            C2 = x_k.shape[1]
            C  = C2 // 2
            magnitude_k = torch.sqrt(
                x_k[:, :C]**2 + x_k[:, C:]**2 + 1e-8
            ).mean(dim=1)   # average over coils → [B, PE, T]

            total_loss = total_loss + self.mse(magnitude_k, target)

        return total_loss / K