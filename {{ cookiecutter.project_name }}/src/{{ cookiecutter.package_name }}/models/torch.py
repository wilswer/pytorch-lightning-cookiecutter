from __future__ import annotations


import torch
import torch.nn as nn

class TorchModel(nn.Module):
    """Torch model."""

    def __init__(self) -> None:
        """Initialize model."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x
