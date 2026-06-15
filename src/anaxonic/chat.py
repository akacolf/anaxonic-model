from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_CHECKPOINT = "runs/hf_chat_quick/best.pt"
DEFAULT_MEMORY_DATA = "data/frontier_chat.jsonl"
DEFAULT_LEARN_DATA = "data/self_learning.jsonl"
DEFAULT_SYSTEM = (
    "You are Anaxonic, a concise local assistant. Answer directly, be accurate, "
    "and say when the available information is not enough instead of guessing."
)


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise SystemExit(
            "Missing dependency: torch\n\n"
            "Use the project virtualenv:\n"
            "  .venv/bin/python -m anaxonic.chat\n\n"
            "Or install the project first:\n"
            '  python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"'
        ) from error
    return torch


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
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.08,
        memory_data: str | None = DEFAULT_MEMORY_DATA,
        learn_data: str | None = DEFAULT_LEARN_DATA,
        generate_unknown: bool = True,
    ):
        torch = require_torch()
        from anaxonic.checkpoint import load_checkpoint, load_model_state
        from anaxonic.config import ModelConfig
        from anaxonic.model import AnaxonicModel
        from anaxonic.responses import ResponseMemory
        from anaxonic.tokenizer import ByteTokenizer
        from anaxonic.train import choose_device

        memory_paths = [memory_data] if memory_data else []
        if learn_data:
            memory_paths.append(learn_data)
        self.memory = ResponseMemory.from_jsonl_files(memory_paths) if memory_paths else None
        self.learn_data = Path(learn_data) if learn_data else None
        self.device = choose_device()
        payload = load_checkpoint(checkpoint_path, self.device)
        model_config = ModelConfig(**payload["config"]["model"])
        self.model = AnaxonicModel(model_config).to(self.device)
        load_model_state(self.model, payload["model"])
        self.model.eval()
        self.tokenizer = ByteTokenizer()
        self.system = system or DEFAULT_SYSTEM
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.generate_unknown = generate_unknown
        self.turns: list[tuple[str, str]] = []
        self.last_user_prompt: str | None = None
        self.last_response: str | None = None
        self.torch = torch

    def teach(self, prompt: str, response: str) -> None:
        if self.learn_data is None:
            raise ValueError("learning log is disabled")
        import json

        self.learn_data.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        with self.learn_data.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.memory is not None:
            self.memory.add(prompt, response)

    def ask(self, prompt: str) -> str:
        from anaxonic.sample import trim_chat_response
        from anaxonic.responses import (
            LOW_CONFIDENCE_RESPONSE,
            built_in_response,
            response_is_low_quality,
        )

        self.turns.append(("user", prompt))
        built_in = built_in_response(prompt)
        if built_in is not None:
            self.turns.append(("assistant", built_in))
            self.last_user_prompt = prompt
            self.last_response = built_in
            return built_in

        if self.memory is not None:
            match = self.memory.lookup(prompt)
            if match is not None:
                self.turns.append(("assistant", match.response))
                self.last_user_prompt = prompt
                self.last_response = match.response
                return match.response
            if not self.generate_unknown:
                self.turns.append(("assistant", LOW_CONFIDENCE_RESPONSE))
                self.last_user_prompt = prompt
                self.last_response = LOW_CONFIDENCE_RESPONSE
                return LOW_CONFIDENCE_RESPONSE

        context = build_chat_context(self.turns, self.system)
        input_ids = self.torch.tensor(
            [self.tokenizer.encode(context)], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )
        output = self.tokenizer.decode(output_ids[0].tolist())
        response = trim_chat_response(output[len(context) :])
        if response_is_low_quality(response):
            response = LOW_CONFIDENCE_RESPONSE
        self.turns.append(("assistant", response))
        self.last_user_prompt = prompt
        self.last_response = response
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to an Anaxonic chat checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--memory-data", default=DEFAULT_MEMORY_DATA)
    parser.add_argument("--learn-data", default=DEFAULT_LEARN_DATA)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-learn-log", action="store_true")
    parser.add_argument(
        "--memory-only",
        action="store_false",
        dest="generate_unknown",
        help="Only answer prompts found in the memory data.",
    )
    parser.set_defaults(generate_unknown=True)
    args = parser.parse_args()

    session = ChatSession(
        args.checkpoint,
        args.system,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        None if args.no_memory else args.memory_data,
        None if args.no_learn_log else args.learn_data,
        args.generate_unknown,
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
        if prompt.startswith("/teach "):
            lesson = prompt.removeprefix("/teach ").strip()
            if "=>" not in lesson:
                print("anaxonic> Use /teach prompt => response")
                continue
            teach_prompt, teach_response = [part.strip() for part in lesson.split("=>", 1)]
            if not teach_prompt or not teach_response:
                print("anaxonic> Use /teach prompt => response")
                continue
            session.teach(teach_prompt, teach_response)
            print("anaxonic> Learned that example for this session and the next training run.")
            continue
        if prompt == "/good":
            if session.last_user_prompt is None or session.last_response is None:
                print("anaxonic> No previous answer to save.")
                continue
            session.teach(session.last_user_prompt, session.last_response)
            print("anaxonic> Saved the last answer as a training example.")
            continue
        if not prompt:
            continue
        print(f"anaxonic> {session.ask(prompt)}")


if __name__ == "__main__":
    main()
