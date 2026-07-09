"""
U-Net Denoiser
==============
A compact residual U-Net used as the CNN denoiser inside each
reconstruction cascade.

Architecture overview:

    Input (2 × in_ch real channels — real & imag stacked)
        │
        ▼
    Encoder  (3 down-sampling blocks: Conv → BN → ReLU × 2)
        │
        ▼
    Bottleneck  (deepest feature map)
        │
        ▼
    Decoder  (3 up-sampling blocks: TransposeConv + skip-cat → Conv × 2)
        │
        ▼
    1×1 Conv  →  Output (2 channels: predicted real & imag)
        │
        + residual connection from input
        ▼
    Final output

The model always operates on real-valued tensors. Complex-to-real and
real-to-complex conversion is handled by the caller (reconstruction_model.py).
"""

"""
U-Net Denoiser — handles arbitrary spatial sizes (e.g. 204x512).
"""

"""
U-Net Denoiser — handles arbitrary spatial sizes (e.g. 204x512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv  = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        x    = self.conv1(x)
        # Crop skip to match x (handles non-power-of-2 sizes like 204)
        skip = skip[:, :, :x.shape[2], :x.shape[3]]
        x    = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Residual U-Net for MRI reconstruction.
    Input/output: real tensor [B, 2*n_coils, H, W].
    Handles ANY spatial size via output pad/crop before residual add.
    """

    def __init__(
        self,
        in_channels:   int = 2,
        out_channels:  int = 2,
        base_features: int = 32,
        n_levels:      int = 3,
    ):
        super().__init__()
        self.n_levels = n_levels

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(ConvBlock(in_channels, base_features))
        ch = base_features
        for _ in range(n_levels - 1):
            self.encoder.append(DownBlock(ch, ch * 2))
            ch *= 2

        # Bottleneck
        self.bottleneck = DownBlock(ch, ch * 2)
        ch *= 2

        # Decoder
        self.decoder = nn.ModuleList()
        feat_list = [base_features * (2 ** i) for i in range(n_levels)]
        for skip_ch in reversed(feat_list):
            self.decoder.append(UpBlock(ch, skip_ch, ch // 2))
            ch //= 2

        self.output_conv     = nn.Conv2d(ch, out_channels, kernel_size=1)
        self.residual_weight = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual    = x
        H_in, W_in = x.shape[2], x.shape[3]

        # Encoder
        skips: List[torch.Tensor] = []
        feat = x
        for enc in self.encoder:
            feat = enc(feat)
            skips.append(feat)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Decoder
        for dec, skip in zip(self.decoder, reversed(skips)):
            feat = dec(feat, skip)

        out = self.output_conv(feat)

        # ── KEY FIX: restore exact input spatial size before residual add ──
        # MaxPool on 204 -> 102 -> 51 -> upsample -> 100 -> 200 != 204
        # We pad the 4 missing rows back so residual + out shapes match.
        H_out, W_out = out.shape[2], out.shape[3]
        if (H_out, W_out) != (H_in, W_in):
            pad_h = H_in - H_out
            pad_w = W_in - W_out
            if pad_h >= 0 and pad_w >= 0:
                out = F.pad(out, (0, pad_w, 0, pad_h))
            else:
                out = out[:, :, :H_in, :W_in]

        return residual + self.residual_weight * out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
