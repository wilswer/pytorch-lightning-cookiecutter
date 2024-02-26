from __future__ import annotations

from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from {{ cookiecutter.package_name }}.models.torch import TorchModel

SheduleSettingType = dict[
    str,
    Union[str, int, torch.optim.lr_scheduler.ReduceLROnPlateau],
]
ScheduleType = Union[torch.optim.Optimizer, SheduleSettingType]
OptimizerType = dict[str, ScheduleType]


class LitModel(pl.LightningModule):
    """Pytorch Lightning model."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        learning_rate_schedule_step_size: int = 10,
        learning_rate_schedule_gamma: float = 0.8,
        reduce_lr_on_plateau_patience: int = 10,
    ) -> None:
        super().__init__()
        self.model = TorchModel()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_rate_schedule_step_size = learning_rate_schedule_step_size
        self.learning_rate_schedule_gamma = learning_rate_schedule_gamma
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.training_step_outputs: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        self.test_step_outputs: list[torch.Tensor] = []
        self.save_hyperparameters()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss function."""
        return nn.functional.mse_loss(y_hat, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def configure_optimizers(
        self,
    ) -> OptimizerType:
        """Configure optimizers with ReduceLROnPlateau."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=self.reduce_lr_on_plateau_patience,
                    min_lr=1e-5,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def step_lr_configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler._LRScheduler],
    ]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.learning_rate_schedule_step_size,
            gamma=self.learning_rate_schedule_gamma,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Take a training step and log loss."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def on_training_epoch_end(self) -> None:
        """End of the training epoch."""
        epoch_avg = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_avg)
        self.training_step_outputs.clear()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Take a validation step and log loss."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """End of the validation epoch."""
        epoch_avg = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_avg)
        self.validation_step_outputs.clear()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Take a test step, log loss and calculate interesting metrics."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, sync_dist=True)
        self.test_step_outputs.append(loss)
        return loss
    
    def on_test_epoch_end(self) -> None:
        """End of the test epoch."""
        epoch_avg = torch.stack(self.test_step_outputs).mean()
        self.log("test_epoch_average", epoch_avg)
        self.test_step_outputs.clear()

    def training_epoch_end(self, outputs: list[StepType]) -> None:  # type: ignore
        """Log average training loss at the end of epoch."""
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, sync_dist=True)

    def validation_epoch_end(self, outputs: list[StepType]) -> None:  # type: ignore
        """Log average validation loss at the end of epoch."""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, sync_dist=True)

    def test_epoch_end(self, outputs: list[StepType]) -> None:  # type: ignore
        """Log average test loss at the end of epoch."""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss, sync_dist=True)