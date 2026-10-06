from __future__ import annotations

import torch
from torch import nn


class XTrustMLP(nn.Module):
    """Lightweight MLP baseline for tabular IoT intrusion-detection data."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
