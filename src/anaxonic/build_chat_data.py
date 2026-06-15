from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from anaxonic.prepare_hf import clean_text, record_to_chat


BAD_SUBSTRINGS = (
    "as an ai language model",
    "i cannot browse",
    "i can't browse",
    "i do not have access to real-time",
    "i don't have access to real-time",
)


def normalize_for_dedupe(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def messages_from_chat(record: Mapping[str, Any]) -> list[dict[str, str]] | None:
    chat = record_to_chat(record)
    if chat is None:
        return None
    if "messages" in chat:
        return list(chat["messages"])
    return [
        {"role": "user", "content": clean_text(chat["prompt"])},
        {"role": "assistant", "content": clean_text(chat["response"])},
    ]


def assistant_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message["role"] == "assistant")


def user_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def is_trainable_chat(
    messages: list[dict[str, str]],
    min_response_chars: int,
    max_response_chars: int,
) -> bool:
    if not any(message["role"] == "user" for message in messages):
        return False
    response = assistant_text(messages).strip()
    if len(response) < min_response_chars or len(response) > max_response_chars:
        return False
    lowered = response.lower()
    return not any(bad in lowered for bad in BAD_SUBSTRINGS)


def canonical_chat_record(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": message["role"], "content": message["content"].strip()}
            for message in messages
            if message["role"] in {"system", "user", "assistant"} and message["content"].strip()
        ]
    }


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        yield payload


def build_chat_dataset(
    inputs: list[str | Path],
    output: str | Path,
    *,
    seed: int = 1337,
    max_examples: int | None = None,
    min_response_chars: int = 2,
    max_response_chars: int = 1200,
) -> int:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for input_path in inputs:
        for record in iter_jsonl(Path(input_path)):
            messages = messages_from_chat(record)
            if messages is None:
                continue
            if not is_trainable_chat(messages, min_response_chars, max_response_chars):
                continue
            key = normalize_for_dedupe(user_text(messages)) + "\n" + normalize_for_dedupe(
                assistant_text(messages)
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(canonical_chat_record(messages))

    random.Random(seed).shuffle(examples)
    if max_examples is not None:
        examples = examples[:max_examples]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned, deduped Anaxonic chat JSONL.")
    parser.add_argument("--input", action="append", required=True, help="Input chat JSONL path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--min-response-chars", type=int, default=2)
    parser.add_argument("--max-response-chars", type=int, default=1200)
    args = parser.parse_args()

    written = build_chat_dataset(
        args.input,
        args.output,
        seed=args.seed,
        max_examples=args.max_examples,
        min_response_chars=args.min_response_chars,
        max_response_chars=args.max_response_chars,
    )
    print(f"wrote {written} cleaned chat examples to {args.output}")


if __name__ == "__main__":
    main()
