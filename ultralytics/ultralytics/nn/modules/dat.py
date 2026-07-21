"""
DAT (Deformable Attention Transformer) integration for YOLOv8.

Implements C2f_DAttention (C2fDAttn): the C2f bottleneck block augmented with
a lightweight Deformable Attention module inside the last bottleneck stage.

Architecture source:
    "Improved YOLOv8 for Road Damage Detection" — Discover Applied Sciences 2024
    (https://doi.org/10.1007/s42452-024-06129-0)

Original attention mechanism:
    Xia et al. "Vision Transformer with Deformable Attention" — CVPR 2022

Integration in backbone:
    Replace C2f at the deeper backbone stages (P4/P5) to enable adaptive,
    deformation-aware feature extraction for irregular road damage patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Bottleneck
from .conv import Conv


class DAttention(nn.Module):
    """
    Deformable Attention module designed for use inside C2f bottleneck blocks.

    Queries are computed from the original feature map; Keys and Values are
    projected from features bilinearly sampled at learned deformed positions.
    Offsets are predicted by a 5x5 depthwise conv → GELU → 1x1 conv sub-network
    (no bias, zero-initialised for training stability — zero offsets == uniform grid).

    The reference grid is optionally downsampled by `grid_stride` (default 2),
    reducing the attention cost from O(N²) to O(N · N/r²).

    Args:
        dim (int): Input channel count.
        num_heads (int): Number of attention heads. Must divide `dim` evenly.
        offset_scale (float): Tanh amplitude for predicted offsets (default 2.0).
        grid_stride (int): Spatial downsampling factor for the reference key grid (default 2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        offset_scale: float = 2.0,
        grid_stride: int = 2,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"DAttention: dim={dim} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.offset_scale = offset_scale
        self.grid_stride = grid_stride

        # ── Offset sub-network ──────────────────────────────────────────────
        # 5×5 depthwise conv: captures local spatial context for offset prediction
        # 1×1 pointwise conv (no bias): outputs 2D offsets; zero-init → start uniform
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1, bias=False),
        )

        # ── Q / K / V / Out projections ─────────────────────────────────────
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        """Zero-init offset network so training begins with a uniform reference grid."""
        nn.init.zeros_(self.offset_net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map from previous bottleneck.
        Returns:
            out: (B, C, H, W) deformable-attention enhanced feature map.
        """
        B, C, H, W = x.shape
        GH = max(1, H // self.grid_stride)
        GW = max(1, W // self.grid_stride)

        # ── 1. Predict spatial offsets ───────────────────────────────────────
        offsets = self.offset_net(x)                        # (B, 2, H, W)
        offsets = offsets.tanh() * self.offset_scale        # clamp to [-s, +s]

        # ── 2. Downsample offsets to reference grid resolution ───────────────
        if GH != H or GW != W:
            offsets_g = F.interpolate(
                offsets, size=(GH, GW), mode="bilinear", align_corners=False
            )
        else:
            offsets_g = offsets

        # Convert pixel offsets → grid coordinate space (normalised to [-1, 1])
        offsets_norm = torch.stack(
            [
                offsets_g[:, 0] / max(W / 2.0, 1.0),
                offsets_g[:, 1] / max(H / 2.0, 1.0),
            ],
            dim=1,
        ).permute(0, 2, 3, 1)  # (B, GH, GW, 2)

        # ── 3. Uniform reference grid in [-1, 1] ─────────────────────────────
        grid_y = torch.linspace(-1 + 1.0 / GH, 1 - 1.0 / GH, GH,
                                device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(-1 + 1.0 / GW, 1 - 1.0 / GW, GW,
                                device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        ref_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, GH, GW, 2)

        # ── 4. Deformed sampling positions ───────────────────────────────────
        deformed_grid = (ref_grid + offsets_norm).clamp(-1.0, 1.0)  # (B, GH, GW, 2)

        # ── 5. Bilinear sample features at deformed positions ────────────────
        x_sampled = F.grid_sample(
            x, deformed_grid, mode="bilinear",
            padding_mode="zeros", align_corners=False
        )  # (B, C, GH, GW)

        # ── 6. Flatten spatial dimensions ────────────────────────────────────
        Nq = H * W
        Nk = GH * GW
        xq = x.permute(0, 2, 3, 1).reshape(B, Nq, C)            # (B, Nq, C)
        xk = x_sampled.permute(0, 2, 3, 1).reshape(B, Nk, C)    # (B, Nk, C)

        # ── 7. Project Q / K / V ─────────────────────────────────────────────
        q = (
            self.q_proj(xq)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Nq, d)
        k = (
            self.k_proj(xk)
            .reshape(B, Nk, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Nk, d)
        v = (
            self.v_proj(xk)
            .reshape(B, Nk, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Nk, d)

        # ── 8. Scaled dot-product attention ──────────────────────────────────
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, Nq, Nk)
        attn = attn.softmax(dim=-1)

        # ── 9. Aggregate and project ──────────────────────────────────────────
        out = (attn @ v)                                    # (B, H, Nq, d)
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
        out = self.out_proj(out)

        # ── 10. Reshape back to (B, C, H, W) ─────────────────────────────────
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class C2fDAttn(nn.Module):
    """
    C2f with Deformable Attention (C2f_DAttention / C2fDAttn).

    Structurally identical to the standard C2f block except that a DAttention
    module is applied — with a residual connection — to the output of the last
    bottleneck before the final channel-merge convolution.

    This is the "C2f_DAttention" block described in:
        "Improved YOLOv8 for Road Damage Detection" — Discover Applied Sciences 2024

    Drop-in replacement for C2f in any YOLOv8 YAML backbone/head definition.
    `num_heads` is auto-derived so that head_dim ≈ 32; no extra YAML argument needed.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of Bottleneck repeats.
        shortcut (bool): Use residual shortcut in Bottleneck.
        g (int): Convolution groups.
        e (float): Channel expansion ratio (default 0.5).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        # Auto-derive num_heads: target head_dim ≈ 32
        num_heads = max(1, self.c // 32)
        # Ensure divisibility
        while self.c % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = DAttention(self.c, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        # Deformable attention with residual on last bottleneck output
        y[-1] = self.attn(y[-1]) + y[-1]
        return self.cv2(torch.cat(y, 1))
