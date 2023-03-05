from __future__ import annotations

import os
from typing import TypedDict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from {{ cookiecutter.package_name }}.data.dataset import TorchDataset


class LoaderArgs(TypedDict, total=False):
    """DataLoader configuration arguments."""

    batch_size: int
    num_workers: int
    pin_memory: bool


class LitDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()
        num_workers = os.cpu_count() if num_workers is None else num_workers
        if num_workers is None:
            raise ValueError("num_workers cannot be None")
        self.loader_args: LoaderArgs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    def setup(self, stage: str | None = None) -> None:
        """Train, validation, and test datasets setup."""
        self.train_dataset = TorchDataset()
        self.train_dataset = TorchDataset()
        self.train_dataset = TorchDataset()

    def train_dataloader(self) -> DataLoader:
        """Train DataLoader."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_args,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_args,
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader."""
        return DataLoader(
            self.test_data,
            shuffle=False,
            **self.loader_args,
        )