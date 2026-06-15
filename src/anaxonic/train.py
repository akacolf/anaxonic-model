from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
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
    parameter_groups = optimizer_parameter_groups(model, weight_decay)
    kwargs = {
        "lr": learning_rate,
    }
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameter_groups, **kwargs)


def optimizer_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    if weight_decay <= 0:
        return [{"params": [param for param in model.parameters()], "weight_decay": 0.0}]

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def configure_torch_runtime(device: torch.device, allow_tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32:
        torch.set_float32_matmul_precision("high")


def autocast_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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


def append_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def dataset_summary(dataset: BatchDataset) -> dict[str, object]:
    examples = getattr(dataset, "examples", None)
    tokens = getattr(dataset, "tokens", None)
    if examples is not None:
        supervised_counts = [int((labels != -100).sum().item()) for _, labels in examples]
        return {
            "kind": "chat",
            "examples": len(examples),
            "supervised_tokens": sum(supervised_counts),
            "avg_supervised_tokens": (
                sum(supervised_counts) / len(supervised_counts) if supervised_counts else 0.0
            ),
        }
    if tokens is not None:
        return {
            "kind": "text",
            "tokens": int(tokens.numel()),
        }
    return {"kind": type(dataset).__name__}


@torch.no_grad()
def estimate_metrics(
    model: AnaxonicModel,
    dataset: BatchDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct_tokens = 0
    supervised_tokens = 0
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
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
    configure_torch_runtime(device, config.train.allow_tf32)
    autocast_enabled = bool(config.train.mixed_precision and device.type == "cuda")
    autocast_dtype = autocast_dtype_for_device(device)
    scaler_enabled = bool(autocast_enabled and autocast_dtype == torch.float16)

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
    if hasattr(torch, "GradScaler"):
        scaler = torch.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / config.train.metrics_file
    if metrics_path.exists():
        metrics_path.unlink()
    train_summary = dataset_summary(train_dataset)
    eval_summary = dataset_summary(eval_dataset)
    random_baseline_loss = math.log(config.model.vocab_size)
    print(
        f"dataset train={train_summary} eval={eval_summary} "
        f"random_baseline_loss={random_baseline_loss:.4f}"
    )
    append_metrics(
        metrics_path,
        {
            "event": "start",
            "run_name": config.run_name,
            "train_dataset": train_summary,
            "eval_dataset": eval_summary,
            "random_baseline_loss": random_baseline_loss,
            "device": device.type,
            "mixed_precision": autocast_enabled,
            "autocast_dtype": str(autocast_dtype).removeprefix("torch."),
            "allow_tf32": bool(config.train.allow_tf32),
        },
    )
    model.train()
    best_eval_loss = float("inf")
    steps_without_improvement = 0
    should_stop = False
    tokens_per_step = (
        config.train.batch_size
        * config.train.gradient_accumulation_steps
        * config.model.context_length
    )
    train_start = time.perf_counter()
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
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("training loss was not computed")
            total_loss += loss.item()
            loss = loss / config.train.gradient_accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        train_loss = total_loss / config.train.gradient_accumulation_steps

        if step == 1 or step % config.train.eval_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - train_start, 1e-9)
            tokens_per_second = (tokens_per_step * step) / elapsed
            eval_loss, eval_accuracy = estimate_metrics(
                model,
                eval_dataset,
                config.train.batch_size,
                config.train.eval_batches,
                device,
                autocast_enabled,
                autocast_dtype,
            )
            train_eval_loss, train_eval_accuracy = estimate_metrics(
                model,
                train_dataset,
                config.train.batch_size,
                config.train.train_eval_batches,
                device,
                autocast_enabled,
                autocast_dtype,
            )
            split_name = "val" if validation_dataset else "eval"
            generalization_gap = eval_loss - train_eval_loss
            print(
                f"step={step} train_loss={train_loss:.4f} "
                f"train_eval_loss={train_eval_loss:.4f} "
                f"{split_name}_loss={eval_loss:.4f} {split_name}_accuracy={eval_accuracy:.3f} "
                f"gap={generalization_gap:.4f} "
                f"ppl={math.exp(min(eval_loss, 20)):.2f} lr={step_lr:.2e} "
                f"grad_norm={grad_norm:.3f} tok/s={tokens_per_second:.0f}"
            )
            append_metrics(
                metrics_path,
                {
                    "event": "eval",
                    "step": step,
                    "train_loss": train_loss,
                    "train_eval_loss": train_eval_loss,
                    "train_eval_accuracy": train_eval_accuracy,
                    f"{split_name}_loss": eval_loss,
                    f"{split_name}_accuracy": eval_accuracy,
                    "generalization_gap": generalization_gap,
                    "learning_rate": step_lr,
                    "grad_norm": grad_norm,
                    "tokens_per_second": tokens_per_second,
                    "perplexity": math.exp(min(eval_loss, 20)),
                    "best_eval_loss": min(best_eval_loss, eval_loss),
                },
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                steps_without_improvement = 0
                save_checkpoint(out_dir / "best.pt", model, optimizer, config, step, train_loss)
            else:
                steps_without_improvement += 1
                if (
                    validation_dataset
                    and config.train.early_stop_patience > 0
                    and steps_without_improvement >= config.train.early_stop_patience
                ):
                    print(
                        "early_stop="
                        f"{steps_without_improvement} evals without validation improvement"
                    )
                    append_metrics(
                        metrics_path,
                        {
                            "event": "early_stop",
                            "step": step,
                            "best_eval_loss": best_eval_loss,
                            "patience": config.train.early_stop_patience,
                        },
                    )
                    should_stop = True

        if should_stop or step % config.train.checkpoint_interval == 0 or step == config.train.max_steps:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, config, step, train_loss)
            save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, config, step, train_loss)
        if should_stop:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["auto", "text", "chat"], default="auto")
    args = parser.parse_args()
    train(args.config, args.data, args.format)


if __name__ == "__main__":
    main()
