from __future__ import annotations

import argparse
import inspect
import math
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

    def split(self, validation_split: float, seed: int) -> tuple["BatchDataset", "BatchDataset | None"]:
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


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def learning_rate_for_step(
    step: int,
    max_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return learning_rate * step / warmup_steps
    if step >= max_steps:
        return min_learning_rate
    decay_start = min(warmup_steps, max_steps)
    decay_steps = max(1, max_steps - decay_start)
    progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (learning_rate - min_learning_rate)


@torch.no_grad()
def estimate_metrics(
    model: AnaxonicModel,
    dataset: BatchDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    autocast_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct_tokens = 0
    supervised_tokens = 0
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
        supervised = y != -100
        if torch.any(supervised):
            predictions = torch.argmax(logits, dim=-1)
            correct_tokens += int((predictions[supervised] == y[supervised]).sum().item())
            supervised_tokens += int(supervised.sum().item())
    model.train()
    accuracy = correct_tokens / supervised_tokens if supervised_tokens else 0.0
    return sum(losses) / max(len(losses), 1), accuracy


def train(config_path: str, data_path: str, data_format: str = "auto") -> None:
    config = load_config(config_path)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device()
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")

    tokenizer = ByteTokenizer()
    dataset = create_dataset(data_path, tokenizer, config.model.context_length, data_format)
    train_dataset, validation_dataset = dataset.split(config.train.validation_split, config.seed)
    eval_dataset = validation_dataset or train_dataset
    model = AnaxonicModel(config.model).to(device)
    if config.train.compile_model:
        model = torch.compile(model)
    optimizer = create_optimizer(
        model,
        config.train.learning_rate,
        config.train.weight_decay,
        device,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    best_eval_loss = float("inf")
    for step in range(1, config.train.max_steps + 1):
        step_lr = learning_rate_for_step(
            step,
            config.train.max_steps,
            config.train.learning_rate,
            config.train.min_learning_rate,
            config.train.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.train.gradient_accumulation_steps):
            x, y = train_dataset.sample_batch(config.train.batch_size, device)
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
            eval_loss, eval_accuracy = estimate_metrics(
                model,
                eval_dataset,
                config.train.batch_size,
                config.train.eval_batches,
                device,
                autocast_enabled,
            )
            split_name = "val" if validation_dataset else "eval"
            print(
                f"step={step} train_loss={train_loss:.4f} "
                f"{split_name}_loss={eval_loss:.4f} {split_name}_accuracy={eval_accuracy:.3f} "
                f"lr={step_lr:.2e}"
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, config, step, train_loss)

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
