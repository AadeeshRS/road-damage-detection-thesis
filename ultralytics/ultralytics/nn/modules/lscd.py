import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Learnable per-scale scalar multiplier used in the LSCD regression branch.

    Per LSCD paper Section 2.2:
        'each detection head uses a shared convolution Conv2d x scale for
         feature scaling, where scale is a scaling factor with an initial value of 1'

    One Scale instance per FPN level (P3, P4, P5); the Conv2d weights are shared.
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ConvGN(nn.Module):
    """
    Conv2d + GroupNorm + SiLU (BN replaced with GN as per the LSCD paper).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int = None,
        g: int = 1,
        groups: int = 16,
        act: bool = True,
    ):
        super().__init__()

        if p is None:
            p = k // 2

        gn_groups = min(groups, c2)
        while c2 % gn_groups != 0:
            gn_groups -= 1

        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.gn = nn.GroupNorm(gn_groups, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class SharedConvBlock(nn.Module):
    """
    ONE shared instance applied to ALL FPN scales (P3, P4, P5).

    Per LSCD paper: 'the three detection heads share two 3x3 convolutional
    modules for feature extraction.'

    Residual connection preserves backbone features.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            ConvGN(channels, channels, k=3, s=1),
            ConvGN(channels, channels, k=3, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LSCDRegressionHead(nn.Module):
    """
    ONE shared Conv2d for regression across all FPN scales.
    Each scale gets its own learnable Scale multiplier.

    Per LSCD paper: 'each detection head uses a shared convolution Conv2d x scale
    for feature scaling, where scale is a scaling factor with an initial value of 1.'

    Call signature: head(shared_feat, scale_idx=i)
    """

    def __init__(self, channels: int, reg_max: int, nl: int):
        """
        Args:
            channels: input channels (c2).
            reg_max:  DFL bins (output = 4 * reg_max channels).
            nl:       number of FPN detection levels (typically 3).
        """
        super().__init__()
        self.shared_conv = nn.Conv2d(channels, 4 * reg_max, kernel_size=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(nl)])

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        return self.scales[scale_idx](self.shared_conv(x))


class LSCDClassificationHead(nn.Module):
    """
    ONE shared Conv2d for classification across all FPN scales.

    Per LSCD paper: 'a shared Conv2d convolution is employed for classification,
    irrespective of the size of the detection layer.'
    """

    def __init__(self, channels: int, nc: int, nl: int):
        super().__init__()
        self.shared_conv = nn.Conv2d(channels, nc, kernel_size=1)
        # Per-scale learnable bias offset — each FPN level needs a different
        # background prior (P3 has 6400 anchors/image, P5 has only 400).
        # Using one global bias (from the shared conv) mis-calibrates P4 and P5.
        self.scale_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(nc)) for _ in range(nl)]
        )

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        return self.shared_conv(x) + self.scale_bias[scale_idx].view(1, -1, 1, 1)
