from __future__ import annotations

import argparse

import torch

from anaxonic.checkpoint import load_checkpoint
from anaxonic.config import ModelConfig
from anaxonic.model import AnaxonicModel
from anaxonic.tokenizer import ByteTokenizer
from anaxonic.train import choose_device


def sample(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    device = choose_device()
    payload = load_checkpoint(checkpoint_path, device)
    model_config = ModelConfig(**payload["config"]["model"])
    model = AnaxonicModel(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(output_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    print(sample(args.checkpoint, args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()

