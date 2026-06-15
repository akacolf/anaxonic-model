from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Protocol

import torch

from anaxonic.checkpoint import save_checkpoint
from anaxonic.config import load_config
from anaxonic.data import ConversationDataset, TextDataset
from anaxonic.model import AnaxonicModel
from anaxonic.tokenizer import ByteTokenizer


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BatchDataset(Protocol):
    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def create_dataset(
    data_path: str,
    tokenizer: ByteTokenizer,
    context_length: int,
    data_format: str,
) -> BatchDataset:
    if data_format == "auto":
        data_format = "chat" if Path(data_path).suffix == ".jsonl" else "text"
    if data_format == "chat":
        return ConversationDataset(data_path, tokenizer, context_length)
    if data_format == "text":
        return TextDataset(data_path, tokenizer, context_length)
    raise ValueError(f"unknown data format: {data_format}")


@torch.no_grad()
def estimate_loss(
    model: AnaxonicModel,
    dataset: BatchDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    autocast_enabled: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(config_path: str, data_path: str, data_format: str = "auto") -> None:
    config = load_config(config_path)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device()
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")

    tokenizer = ByteTokenizer()
    dataset = create_dataset(data_path, tokenizer, config.model.context_length, data_format)
    model = AnaxonicModel(config.model).to(device)
    if config.train.compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(1, config.train.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.train.gradient_accumulation_steps):
            x, y = dataset.sample_batch(config.train.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("training loss was not computed")
            total_loss += loss.item()
            loss = loss / config.train.gradient_accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss = total_loss / config.train.gradient_accumulation_steps

        if step == 1 or step % config.train.eval_interval == 0:
            eval_loss = estimate_loss(
                model,
                dataset,
                config.train.batch_size,
                config.train.eval_batches,
                device,
                autocast_enabled,
            )
            print(f"step={step} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        if step % config.train.checkpoint_interval == 0 or step == config.train.max_steps:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, config, step, train_loss)
            save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, config, step, train_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["auto", "text", "chat"], default="auto")
    args = parser.parse_args()
    train(args.config, args.data, args.format)


if __name__ == "__main__":
    main()
