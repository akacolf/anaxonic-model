from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.1

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 250
    out_dir: str = "runs/tiny"
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        run_name=str(payload.get("run_name", "anaxonic")),
        seed=int(payload.get("seed", 1337)),
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "model": config.model.__dict__,
        "train": config.train.__dict__,
    }
