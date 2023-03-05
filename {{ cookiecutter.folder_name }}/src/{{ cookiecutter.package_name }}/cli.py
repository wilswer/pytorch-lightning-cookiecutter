from __future__ import annotations

import os
from typing import Optional

import typer

from {{ cookiecutter.package_name }}.train.trainer import train


app = typer.Typer(help="Train and evaluate auotregressive models")


N_GPUS = typer.Option(7, min=0, max=7, help="Number of GPUs to use")
N_EPOCHS = typer.Option(10, min=1, help="Number of epochs to train for")
BATCH_SIZE = typer.Option(64, min=1, help="Batch size")
LEARNING_RATE = typer.Option(0.001, min=0.0, max=1.0, help="Learning rate")
DEBUG = typer.Option(False, help="Debug mode")


@app.command()
def train(
    gpus: int = N_GPUS,
    epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    debug: bool = DEBUG,
) -> None:
    """Train an autoregressive model.
    Args:
        gpus (int): Number of GPUs to use. Defaults to 7.
        epochs (int): Number of epochs to train. Defaults to 10.
        batch_size (int): Training mini-batch size. Defaults to 64.
        learning_rate (float): Learning rate. Defaults to 0.001.
        debug (bool): Run in debug mode. Defaults to False.
    Raises
    ------
        ValueError: If model type is not recognized.
    """
    train(
        n_gpus=gpus,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        debug=debug,
    )

MODEL_PATH = typer.Option(None, help="Path to model checkpoint")


@app.command()
def evaluate(
    model_path: Optional[str] = MODEL_PATH,
) -> None:
    """Evaluate an autoregressive model.
    Args:
        model_path (str, optional): Path to model. Defaults to None.
    Raises
    ------
        ValueError: If model path is not found.
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "model.pth"
        )
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} not found")


if __name__ == "__main__":
    app()