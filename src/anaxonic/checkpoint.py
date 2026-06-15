from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from anaxonic.config import ExperimentConfig, config_to_dict
from anaxonic.model import AnaxonicModel

LEGACY_NON_PARAMETER_KEYS = (".attn.causal_mask",)


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
    checkpoint_model = getattr(model, "_orig_mod", model)
    payload = {
        "model": checkpoint_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_to_dict(config),
        "step": step,
        "train_loss": train_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)


def normalize_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state_dict.items():
        if key.endswith(LEGACY_NON_PARAMETER_KEYS):
            continue
        normalized[key.removeprefix("_orig_mod.")] = value
    return normalized


def load_model_state(model: AnaxonicModel, state_dict: dict[str, Any]) -> None:
    model.load_state_dict(normalize_model_state_dict(state_dict))
