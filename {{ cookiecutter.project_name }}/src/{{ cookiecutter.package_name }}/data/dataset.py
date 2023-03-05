from __future__ import annotations


import torch
from torch.utils.data import Dataset

from {{ cookiecutter.package_name }}.utils.utilities import get_data


class TorchDataset(Dataset):
    """PyTorch Dataset."""

    def __init__(self) -> None:
        """Initialize dataset."""
        self.data = get_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.data.get_x()
        y = self.data.get_y()
        return X, y
