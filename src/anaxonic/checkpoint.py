from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from anaxonic.config import ExperimentConfig, config_to_dict
from anaxonic.model import AnaxonicModel


def save_checkpoint(
    path: str | Path,
    model: AnaxonicModel,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
    train_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_to_dict(config),
        "step": step,
        "train_loss": train_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)

