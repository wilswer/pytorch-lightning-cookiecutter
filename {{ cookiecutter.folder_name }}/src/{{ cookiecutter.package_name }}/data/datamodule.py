from __future__ import annotations

import os
from typing import TypedDict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

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
        seed: int | None = None,
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
        self.dataset = TorchDataset()
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        """Train, validation, and test datasets setup."""
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )

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
            self.test_dataset,
            shuffle=False,
            **self.loader_args,
        )
