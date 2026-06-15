from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "user": "user",
}

PROMPT_RESPONSE_FIELDS = (
    ("prompt", "response"),
    ("instruction", "output"),
    ("question", "answer"),
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def combined_length(record: Mapping[str, Any]) -> int:
    if "messages" in record:
        return sum(len(clean_text(message.get("content"))) for message in record["messages"])
    return len(clean_text(record.get("prompt"))) + len(clean_text(record.get("response")))


def normalize_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw_role = clean_text(item.get("role", item.get("from"))).lower()
        role = ROLE_ALIASES.get(raw_role)
        content = clean_text(item.get("content", item.get("value")))
        if role and content:
            messages.append({"role": role, "content": content})
    if any(message["role"] == "assistant" for message in messages):
        return messages
    return None


def detect_prompt_response_fields(record: Mapping[str, Any]) -> tuple[str, str] | None:
    for prompt_field, response_field in PROMPT_RESPONSE_FIELDS:
        if clean_text(record.get(prompt_field)) and clean_text(record.get(response_field)):
            return prompt_field, response_field
    return None


def record_to_chat(
    record: Mapping[str, Any],
    prompt_field: str | None = None,
    response_field: str | None = None,
    input_field: str | None = "input",
    messages_field: str | None = None,
) -> dict[str, Any] | None:
    if messages_field and messages_field in record:
        messages = normalize_messages(record[messages_field])
        if messages:
            return {"messages": messages}

    for candidate in ("messages", "conversations"):
        if candidate in record:
            messages = normalize_messages(record[candidate])
            if messages:
                return {"messages": messages}

    fields = None
    if prompt_field and response_field:
        fields = (prompt_field, response_field)
    else:
        fields = detect_prompt_response_fields(record)
    if fields is None:
        return None

    prompt = clean_text(record.get(fields[0]))
    response = clean_text(record.get(fields[1]))
    extra_input = clean_text(record.get(input_field)) if input_field else ""
    if extra_input:
        prompt = f"{prompt}\n\n{extra_input}" if prompt else extra_input
    if not prompt or not response:
        return None
    return {"prompt": prompt, "response": response}


def record_to_text(record: Mapping[str, Any], text_field: str) -> str | None:
    text = clean_text(record.get(text_field))
    return text or None


def keep_record(record: Mapping[str, Any], min_chars: int, max_chars: int | None) -> bool:
    length = combined_length(record)
    if length < min_chars:
        return False
    return max_chars is None or length <= max_chars


def iter_limited(records: Iterable[Mapping[str, Any]], max_examples: int | None):
    kept = 0
    for record in records:
        yield record
        kept += 1
        if max_examples is not None and kept >= max_examples:
            return


def load_hf_dataset(args: argparse.Namespace) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face datasets package is required. "
            'Install it with: pip install -e ".[data]"'
        ) from exc

    kwargs: dict[str, Any] = {
        "path": args.dataset,
        "name": args.subset,
        "split": args.split,
        "streaming": args.streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    return load_dataset(**{key: value for key, value in kwargs.items() if value is not None})


def write_converted_dataset(records: Iterable[Mapping[str, Any]], args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for source_record in records:
            if args.format == "chat":
                record = record_to_chat(
                    source_record,
                    prompt_field=args.prompt_field,
                    response_field=args.response_field,
                    input_field=args.input_field,
                    messages_field=args.messages_field,
                )
                if record is None or not keep_record(record, args.min_chars, args.max_chars):
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                text = record_to_text(source_record, args.text_field)
                if text is None:
                    continue
                if len(text) < args.min_chars or (args.max_chars is not None and len(text) > args.max_chars):
                    continue
                handle.write(text + "\n\n")
            written += 1
            if args.max_examples is not None and written >= args.max_examples:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face dataset split into Anaxonic training data."
    )
    parser.add_argument("--dataset", required=True, help="Dataset id or local dataset path.")
    parser.add_argument("--subset", help="Dataset subset/config name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["chat", "text"], default="chat")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--messages-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--response-field")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    records = load_hf_dataset(args)
    written = write_converted_dataset(records, args)
    print(f"wrote {written} {args.format} examples to {args.output}")


if __name__ == "__main__":
    main()
