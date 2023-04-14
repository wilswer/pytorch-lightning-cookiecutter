from __future__ import annotations

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from {{ cookiecutter.package_name }}.models.lightning import LitModel
from {{ cookiecutter.package_name }}.data.datamodule import LitDataModule


def train(
    n_gpus: int = 7,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    debug: bool = False,
) -> None:
    """Train the lightning model.

    Args:
        n_gpus (int, optional): Number of GPUs to use. Defaults to 7.
        n_epochs (int, optional): Number of epochs to train. Defaults to 100.
        batch_size (int, optional): Mini-batch size. Defaults to 64.
        learning_rate (float, optional): Training learning rate. Defaults to 0.001.
        debug (bool, optional): Run in debug mode. Defaults to False.

    Raises
    ------
        ValueError: If the model directory does not exist.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist.")
    model_checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        filename="model_name-{epoch:003d}-{val_loss:.5f}-{step}",
        save_top_k=5,
        monitor="train_loss",
        save_last=True,
        auto_insert_metric_name=True,
    )
    model = LitModel(
        lr=learning_rate,
        batch_size=batch_size,
    )
    datamodule = LitDataModule(
        batch_size=batch_size,
    )
    if debug:
        trainer = pl.Trainer(fast_dev_run=True, gpus=None)
    if n_gpus == 1:
        trainer = pl.Trainer(
            default_root_dir=model_dir,
            accelerator="gpu",
            devices=n_gpus,
            max_epochs=n_epochs,
            callbacks=[lr_monitor, model_checkpoint],
        )
    elif n_gpus > 1:
        trainer = pl.Trainer(
            default_root_dir=model_dir,
            accelerator="gpu",
            devices=n_gpus,
            max_epochs=n_epochs,
            strategy="ddp_find_unused_parameters_false",
            callbacks=[lr_monitor, model_checkpoint],
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=model_dir,
            accelerator="cpu",
            max_epochs=n_epochs,
            callbacks=[lr_monitor, model_checkpoint],
        )

    trainer.fit(model=model, datamodule=datamodule)

