"""
AE-FPN (Adaptive EMA Feature Pyramid Network) modules for YOLOv8.

Implements two attention modules that replace/augment the standard FPN-PAN neck:

    EMA  — Efficient Multi-scale Attention gate inserted after each FPN fusion.
             Parallel 1×1 + 3×3 branches with directional pooling for spatial-channel
             attention, boosting feature retention at every scale junction.

    AAM  — Adaptive Attention Module (Coordinate Attention) applied to the deepest
             backbone feature (P5/SPPF output) before it enters the FPN top-down path,
             enhancing large-scale target sensitivity without losing location cues.

Architecture source:
    SEA-YOLO: "Detection of Road Damage Based on SEA-YOLO"
    PLOS ONE https://doi.org/10.1371/journal.pone.0324439, June 2025.

EMA attention mechanism inspired by:
    Coordinate Attention (Hou et al., CVPR 2021).
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention (EMA) gate.

    Placed after each FPN / PANet C2f fusion block in the AE-FPN neck.
    Suppresses noisy fused features and emphasises spatially important channels
    via a parallel 1×1 / 3×3-depthwise-conv structure with directional strip pooling.

    Shape: (B, C, H, W) → (B, C, H, W)  [channel-preserving]

    Args:
        c1 (int): Input (= output) channel count. Inferred from YAML by parse_model.
    """

    def __init__(self, c1: int):
        super().__init__()
        mid = max(8, c1 // 8)

        # ── 1×1 branch: channel interaction ─────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # ── 3×3 depth-wise branch: local spatial feature extraction ─────────
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # ── Directional strip pooling (H-pool + W-pool → encode → decode) ───
        self.encode = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.SiLU(),
        )
        self.decode_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.decode_w = nn.Conv2d(mid, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Two parallel feature branches
        f1 = self.conv1(x)   # channel mixing
        f3 = self.conv3(x)   # local spatial

        # ── Directional pooling from 1×1 branch ─────────────────────────────
        # Strip-pool along each spatial axis
        xh = f1.mean(dim=-1, keepdim=True)                           # (B,C,H,1)
        xw = f1.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)      # (B,C,W,1)

        # Concat H and W pooled features along spatial dim, then encode
        y = torch.cat([xh, xw], dim=2)   # (B, C, H+W, 1)
        y = self.encode(y)                # (B, mid, H+W, 1)

        # Split back and decode to attention weights
        y_h, y_w = y.split([H, W], dim=2)
        attn_h = self.sigmoid(self.decode_h(y_h))                     # (B,C,H,1)
        attn_w = self.sigmoid(self.decode_w(y_w)).permute(0, 1, 3, 2) # (B,C,1,W)

        # Apply attention to 3×3 branch, add 1×1 branch as residual
        return f3 * attn_h * attn_w + f1


class AAM(nn.Module):
    """
    Adaptive Attention Module (AAM) based on Coordinate Attention.

    Applied exclusively to the deepest backbone feature (SPPF / P5 output) in
    AE-FPN before the top-down FPN path begins.  Preserves precise 2-D location
    information while suppressing irrelevant channels, improving large-crack and
    pothole detection sensitivity.

    Shape: (B, C, H, W) → (B, C, H, W)  [channel-preserving]

    Args:
        c1 (int): Input (= output) channel count. Inferred from YAML by parse_model.
        reduction (int): Channel reduction ratio for the shared encoding bottleneck.
    """

    def __init__(self, c1: int, reduction: int = 8):
        super().__init__()
        mid = max(8, c1 // reduction)

        # Shared encoding of H + W pooled context
        self.encode = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )

        # Separate decode heads for vertical and horizontal attention maps
        self.decode_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.decode_w = nn.Conv2d(mid, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Strip pooling along H and W axes ────────────────────────────────
        xh = x.mean(dim=-1, keepdim=True)                            # (B,C,H,1)
        xw = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)       # (B,C,W,1)

        # Encode joint spatial context
        y = torch.cat([xh, xw], dim=2)    # (B, C, H+W, 1)
        y = self.encode(y)                 # (B, mid, H+W, 1)

        # Decode directional attention weights
        y_h, y_w = y.split([H, W], dim=2)
        attn_h = self.sigmoid(self.decode_h(y_h))                      # (B,C,H,1)
        attn_w = self.sigmoid(self.decode_w(y_w)).permute(0, 1, 3, 2)  # (B,C,1,W)

        # Element-wise gating: preserve location-aware channel emphasis
        return x * attn_h * attn_w
