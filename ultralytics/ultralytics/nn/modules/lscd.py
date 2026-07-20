import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Learnable scale parameter used in the LSCD regression branch.
    """

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class ConvGN(nn.Module):
    """
    Conv2d + GroupNorm + SiLU
    """

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        groups=16,
        act=True,
    ):
        super().__init__()

        if p is None:
            p = k // 2

        # Ensure GroupNorm group count divides channels
        gn_groups = min(groups, c2)
        while c2 % gn_groups != 0:
            gn_groups -= 1

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False,
        )

        self.gn = nn.GroupNorm(gn_groups, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))