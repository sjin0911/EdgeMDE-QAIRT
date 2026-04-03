from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Minimal convolutional network for export and device pipeline smoke tests."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
