# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P2-Proxy Guided Fine-Grained Vehicle Recovery Pyramid (PVRP) modules.

Three complementary sub-modules for UAV vehicle detection. Each is exposed as a
drop-in YAML component for YOLO11 and is independently togglable by swapping
model config files:

1. ``P2Proxy``               -- lightweight proxy branch preserving P2/4 details
2. ``AntiAliasDown``         -- anti-aliased stride-2 downsampling (blur+conv)
3. ``SemanticGatedFuse``     -- high-level semantics gate the proxy to suppress
                                background when it is fused into P3/8
4. ``NeighborDecoupleAdapter`` -- local-contrast adapter that only feeds the P3
                                  detection head, sharpening boundaries between
                                  near-neighbor vehicles in dense scenes
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = "P2Proxy", "AntiAliasDown", "SemanticGatedFuse", "NeighborDecoupleAdapter"


class P2Proxy(nn.Module):
    """Lightweight C3k2-Lite proxy block for P2/4 features.

    Follows the C2f / C3k2 split-transform-merge pattern but uses depthwise
    separable inner transforms and a modest expansion ratio to stay within the
    parameter budget at P2/4 resolution (4x the pixel count of P3/8).

    Args:
        c1 (int): Input channel count.
        c2 (int): Output channel count.
        n (int): Number of inner DWConv+1x1 transform branches.
        shortcut (bool): Whether each inner branch uses a residual shortcut.
        e (float): Hidden channel expansion ratio, typically <=0.5 for Lite.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5) -> None:
        super().__init__()
        self.c = max(int(c2 * e), 8)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(DWConv(self.c, self.c, 3), Conv(self.c, self.c, 1))
            for _ in range(n)
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            out = m(y[-1])
            if self.add:
                out = out + y[-1]
            y.append(out)
        return self.cv2(torch.cat(y, 1))


class AntiAliasDown(nn.Module):
    """Anti-aliased stride-2 downsampler.

    Applies a fixed low-pass Binomial filter (depthwise, untrained) before a
    stride-2 3x3 convolution. Adapted from Zhang 2019 "Making Convolutional
    Networks Shift-Invariant Again". For small UAV vehicles this reduces the
    high-frequency aliasing that accompanies a direct stride-2 conv when
    squeezing P2/4 features down to P3/8 resolution.

    Args:
        c1 (int): Input channel count.
        c2 (int): Output channel count.
        blur_k (int): Low-pass kernel size, must be 3 or 5.
    """

    def __init__(self, c1: int, c2: int, blur_k: int = 3) -> None:
        super().__init__()
        if blur_k == 3:
            kernel = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        elif blur_k == 5:
            kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        else:
            raise ValueError(f"blur_k must be 3 or 5, got {blur_k}")
        kernel = (kernel[:, None] * kernel[None, :]).view(1, 1, blur_k, blur_k).repeat(c1, 1, 1, 1)
        self.register_buffer("blur_weight", kernel, persistent=False)
        self.c1 = c1
        self.blur_pad = (blur_k - 1) // 2
        self.conv = Conv(c1, c2, k=3, s=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.blur_weight, stride=1, padding=self.blur_pad, groups=self.c1)
        return self.conv(x)


class SemanticGatedFuse(nn.Module):
    """Semantic-gated fusion of a low-level proxy and a high-level neck feature.

    The high-level feature is reduced and drives two gates:
        * a channel gate (global-pool + 1x1) that re-weights proxy channels;
        * a spatial gate (1x1) that re-weights proxy positions.
    The gated proxy is concatenated with the high-level feature, then a 1x1
    convolution compresses the concatenation to ``c_out`` channels.

    Args:
        c_low (int): Channels of the low-level (proxy) input.
        c_high (int): Channels of the high-level (neck) input.
        c_out (int): Output channel count.
        reduction (int): Reduction ratio for the gate network bottleneck.

    Forward input:
        x (list[Tensor, Tensor]): ``[proxy_low, neck_high]``; both tensors are
        expected at the same spatial resolution (i.e. P3/8 for PVRP).
    """

    def __init__(self, c_low: int, c_high: int, c_out: int, reduction: int = 4) -> None:
        super().__init__()
        c_mid = max(c_high // reduction, 16)
        self.gate_conv = Conv(c_high, c_mid, k=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid, c_low, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c_low + c_high, c_out, k=1)

    def forward(self, x):
        low, high = x
        g = self.gate_conv(high)
        ch_g = self.channel_gate(g)
        sp_g = self.spatial_gate(g)
        low_gated = low * ch_g * sp_g
        return self.fuse(torch.cat([low_gated, high], dim=1))


class NeighborDecoupleAdapter(nn.Module):
    """Neighbor-decoupling adapter for the P3 detection head input.

    Uses a Difference-of-Gaussians-style local contrast gate to strengthen
    boundary responses between near-neighbor vehicles in dense scenes. Runs at
    P3/8 resolution only. The gated residual ``x * (1 + gate)`` preserves the
    original feature when the gate is silent, making it safe as a late adapter.

    Args:
        c1 (int): Input channel count.
        c2 (int): Output channel count (usually equal to ``c1``).
        reduction (int): Reduction ratio for the contrast branch.
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4) -> None:
        super().__init__()
        c_mid = max(c1 // reduction, 16)
        self.cv_red = Conv(c1, c_mid, k=1)
        self.cv_local = Conv(c_mid, c_mid, k=3)
        self.cv_context = Conv(c_mid, c_mid, k=5)
        self.contrast_gate = nn.Sequential(
            nn.Conv2d(c_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.cv_out = Conv(c1, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cv_red(x)
        local = self.cv_local(f)
        context = self.cv_context(f)
        gate = self.contrast_gate(local - context)
        return self.cv_out(x * (1.0 + gate))
