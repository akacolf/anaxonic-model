from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from anaxonic.checkpoint import save_checkpoint
from anaxonic.config import load_config
from anaxonic.data import TextDataset
from anaxonic.model import AnaxonicModel
from anaxonic.tokenizer import ByteTokenizer


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: AnaxonicModel,
    dataset: TextDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(config_path: str, data_path: str) -> None:
    config = load_config(config_path)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device()

    tokenizer = ByteTokenizer()
    dataset = TextDataset(data_path, tokenizer, config.model.context_length)
    model = AnaxonicModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(1, config.train.max_steps + 1):
        x, y = dataset.sample_batch(config.train.batch_size, device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("training loss was not computed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if step == 1 or step % config.train.eval_interval == 0:
            eval_loss = estimate_loss(
                model,
                dataset,
                config.train.batch_size,
                config.train.eval_batches,
                device,
            )
            print(f"step={step} train_loss={loss.item():.4f} eval_loss={eval_loss:.4f}")

        if step % config.train.checkpoint_interval == 0 or step == config.train.max_steps:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, config, step, loss.item())
            save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, config, step, loss.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args.config, args.data)


if __name__ == "__main__":
    main()

