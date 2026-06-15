from __future__ import annotations

import argparse

import torch

from anaxonic.checkpoint import load_checkpoint
from anaxonic.config import ModelConfig
from anaxonic.model import AnaxonicModel
from anaxonic.responses import (
    DEFAULT_RESPONSE_DATA,
    LOW_CONFIDENCE_RESPONSE,
    ResponseMemory,
    response_is_low_quality,
)
from anaxonic.tokenizer import ByteTokenizer
from anaxonic.train import choose_device


def format_chat_prompt(prompt: str, system: str | None = None) -> str:
    parts = []
    if system:
        parts.append(f"<|system|>\n{system.strip()}\n")
    parts.append(f"<|user|>\n{prompt.strip()}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def trim_chat_response(output: str) -> str:
    for marker in ("\n<|user|>", "\n<|system|>", "\n<|assistant|>"):
        if marker in output:
            output = output.split(marker, 1)[0]
    return output.split("\n", 1)[0].strip()


def sample(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    chat: bool = False,
    system: str | None = None,
    memory_data: str | None = DEFAULT_RESPONSE_DATA,
    generate_unknown: bool = False,
) -> str:
    if chat and memory_data:
        match = ResponseMemory.from_jsonl(memory_data).lookup(prompt)
        if match is not None:
            return match.response
        if not generate_unknown:
            return LOW_CONFIDENCE_RESPONSE

    device = choose_device()
    payload = load_checkpoint(checkpoint_path, device)
    model_config = ModelConfig(**payload["config"]["model"])
    model = AnaxonicModel(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    if chat:
        prompt = format_chat_prompt(prompt, system)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
    output = tokenizer.decode(output_ids[0].tolist())
    if chat:
        response = trim_chat_response(output[len(prompt) :])
        if response_is_low_quality(response):
            return LOW_CONFIDENCE_RESPONSE
        return response
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--memory-data", default=DEFAULT_RESPONSE_DATA)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--generate-unknown", action="store_true")
    args = parser.parse_args()
    print(
        sample(
            args.checkpoint,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.chat,
            args.system,
            None if args.no_memory else args.memory_data,
            args.generate_unknown,
        )
    )


if __name__ == "__main__":
    main()
