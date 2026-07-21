"""Dynamic Efficient Multi-scale Attention (D-EMA)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _largest_divisor_not_above(value: int, upper: int) -> int:
    """Return the largest divisor of ``value`` not larger than ``upper``."""
    for divisor in range(min(value, upper), 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


class D_EMA(nn.Module):
    r"""Paper-aligned D-EMA block.

    The nominal group count follows equation (2):

    .. math:: G = \min(G_{max}, \lfloor C/T_c \rfloor)

    For engineering robustness, if that value does not divide ``C``, the
    nearest smaller divisor is used so channel groups remain equal-sized.

    The spatial branch creates a single ``H x W`` attention map per group from
    horizontal and vertical pooled descriptors. The local branch is
    ``Conv3x3 + BN + ReLU``. Fusion follows equations (9)-(11):

    ``refined = local_feature * spatial_attention``
    ``enhanced = refined + alpha * input_group``
    """

    def __init__(
        self,
        channels: int,
        G_max: int = 8,
        T_c: int = 4,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or G_max <= 0 or T_c <= 0:
            raise ValueError("channels, G_max and T_c must be positive")

        nominal_groups = max(1, min(G_max, channels // T_c))
        self.channels = int(channels)
        self.groups = _largest_divisor_not_above(self.channels, nominal_groups)
        self.group_channels = self.channels // self.groups

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # One spatial descriptor per channel group, matching A_s in R^(1xHxW).
        self.spatial_fuse = nn.Conv2d(self.group_channels, 1, kernel_size=1, bias=True)
        self.local_conv = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.local_bn = nn.BatchNorm2d(self.group_channels)
        self.local_act = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"D_EMA expects BCHW input, got shape {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"D_EMA was built for {self.channels} channels but received {channels}. "
                "Pass the actual input channel count from parse_model."
            )

        grouped = x.reshape(batch * self.groups, self.group_channels, height, width)

        # Spatial-aware branch: directional pooling -> fusion -> sigmoid ->
        # two-dimensional spatial softmax.
        horizontal = self.pool_h(grouped)  # [BG, Cg, H, 1]
        vertical = self.pool_w(grouped).transpose(2, 3)  # [BG, Cg, W, 1]
        directional = torch.cat((horizontal, vertical), dim=2)
        directional = torch.sigmoid(self.spatial_fuse(directional))
        horizontal_logits, vertical_logits = torch.split(
            directional, (height, width), dim=2
        )
        vertical_logits = vertical_logits.transpose(2, 3)
        spatial_logits = horizontal_logits + vertical_logits  # [BG, 1, H, W]
        spatial_attention = torch.softmax(spatial_logits.flatten(2), dim=-1).view(
            batch * self.groups, 1, height, width
        )

        # Local-context branch and equation (9).
        local_feature = self.local_act(self.local_bn(self.local_conv(grouped)))
        refined = local_feature * spatial_attention

        # Equations (10)-(11).
        enhanced = refined + self.alpha * grouped
        return enhanced.reshape(batch, channels, height, width)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, groups={self.groups}, "
            f"group_channels={self.group_channels}"
        )
