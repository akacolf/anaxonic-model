from __future__ import annotations

import argparse

import torch

from anaxonic.checkpoint import load_checkpoint
from anaxonic.config import ModelConfig
from anaxonic.model import AnaxonicModel
from anaxonic.sample import trim_chat_response
from anaxonic.tokenizer import ByteTokenizer
from anaxonic.train import choose_device


def build_chat_context(turns: list[tuple[str, str]], system: str | None = None) -> str:
    parts = []
    if system:
        parts.append(f"<|system|>\n{system.strip()}\n")
    for role, content in turns:
        parts.append(f"<|{role}|>\n{content.strip()}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


class ChatSession:
    def __init__(
        self,
        checkpoint_path: str,
        system: str | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ):
        self.device = choose_device()
        payload = load_checkpoint(checkpoint_path, self.device)
        model_config = ModelConfig(**payload["config"]["model"])
        self.model = AnaxonicModel(model_config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.tokenizer = ByteTokenizer()
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.turns: list[tuple[str, str]] = []

    @torch.no_grad()
    def ask(self, prompt: str) -> str:
        self.turns.append(("user", prompt))
        context = build_chat_context(self.turns, self.system)
        input_ids = torch.tensor([self.tokenizer.encode(context)], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_ids,
            self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        output = self.tokenizer.decode(output_ids[0].tolist())
        response = trim_chat_response(output[len(context) :])
        self.turns.append(("assistant", response))
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to an Anaxonic chat checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    session = ChatSession(
        args.checkpoint,
        args.system,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    print("Anaxonic chat. Type /exit to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            break
        if prompt in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        print(f"anaxonic> {session.ask(prompt)}")


if __name__ == "__main__":
    main()
