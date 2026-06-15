from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_RESPONSE_DATA = "data/basic_responses.jsonl"
LOW_CONFIDENCE_RESPONSE = (
    "I am ready to help. Please send the question again with a little more detail."
)


def normalize_prompt(prompt: str) -> str:
    text = prompt.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class ResponseMatch:
    prompt: str
    response: str
    score: float


class ResponseMemory:
    def __init__(self, examples: list[tuple[str, str]]):
        self.examples = [(prompt, response, normalize_prompt(prompt)) for prompt, response in examples]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ResponseMemory":
        examples: list[tuple[str, str]] = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.extend(_examples_from_record(payload, line_number, path))
        return cls(examples)

    def lookup(self, prompt: str, min_score: float = 0.74) -> ResponseMatch | None:
        normalized = normalize_prompt(prompt)
        if not normalized:
            return None

        best: ResponseMatch | None = None
        for source_prompt, response, source_normalized in self.examples:
            if normalized == source_normalized:
                return ResponseMatch(source_prompt, response, 1.0)
            score = SequenceMatcher(None, normalized, source_normalized).ratio()
            if best is None or score > best.score:
                best = ResponseMatch(source_prompt, response, score)

        if best is None or best.score < min_score:
            return None
        return best


def _examples_from_record(
    payload: dict[str, Any], line_number: int, path: str | Path
) -> list[tuple[str, str]]:
    if "prompt" in payload and "response" in payload:
        return [(str(payload["prompt"]), str(payload["response"]))]
    if "messages" not in payload:
        raise ValueError(
            f"{path}:{line_number} must contain either messages or prompt/response fields"
        )

    messages = payload["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"{path}:{line_number} messages must be a list")

    examples = []
    last_user: str | None = None
    for message in messages:
        role = str(message["role"]).strip().lower()
        content = str(message["content"])
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user is not None:
            examples.append((last_user, content))
            last_user = None
    return examples


def response_is_low_quality(response: str) -> bool:
    cleaned = response.strip()
    if not cleaned:
        return True
    if "�" in cleaned:
        return True

    words = re.findall(r"[A-Za-z]+", cleaned)
    if len(cleaned) < 12 and len(words) <= 2:
        return True
    if len(words) >= 2 and len(set(word.casefold() for word in words)) == 1:
        return True
    if len(words) >= 12:
        folded_words = [word.casefold() for word in words]
        short_word_ratio = sum(len(word) <= 4 for word in folded_words) / len(folded_words)
        unique_word_ratio = len(set(folded_words)) / len(folded_words)
        if short_word_ratio > 0.75 and unique_word_ratio < 0.65:
            return True
    if re.fullmatch(r"(ye|yes?)\s+(an|ans|d|s|t)[\w\s.]*", cleaned.casefold()):
        return True
    return False
