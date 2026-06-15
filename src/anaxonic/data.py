from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from anaxonic.tokenizer import ByteTokenizer


class TextDataset:
    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, context_length: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        text = self.path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= context_length:
            raise ValueError(
                f"{self.path} has {len(token_ids)} tokens, but needs more than {context_length}"
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        max_start = len(self.tokens) - self.context_length - 1
        starts = torch.randint(0, max_start + 1, (batch_size,))
        x = torch.stack([self.tokens[i : i + self.context_length] for i in starts])
        y = torch.stack([self.tokens[i + 1 : i + self.context_length + 1] for i in starts])
        return x.to(device), y.to(device)


class ConversationDataset:
    """JSONL instruction dataset with response-only loss targets."""

    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, context_length: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples = self._load_examples()
        if not self.examples:
            raise ValueError(f"{self.path} did not contain any trainable assistant responses")

    def _load_examples(self) -> list[tuple[Tensor, Tensor]]:
        examples = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = self._messages_from_record(payload, line_number)
            token_ids, labels = self._encode_messages(messages)
            if any(label != -100 for label in labels):
                examples.append(self._pack_example(token_ids, labels))
        return examples

    def _messages_from_record(self, payload: dict[str, Any], line_number: int) -> list[dict[str, str]]:
        if "messages" in payload:
            messages = payload["messages"]
            if not isinstance(messages, list):
                raise ValueError(f"{self.path}:{line_number} messages must be a list")
            return [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in messages
            ]
        if "prompt" in payload and "response" in payload:
            return [
                {"role": "user", "content": str(payload["prompt"])},
                {"role": "assistant", "content": str(payload["response"])},
            ]
        raise ValueError(
            f"{self.path}:{line_number} must contain either messages or prompt/response fields"
        )

    def _encode_messages(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        token_ids: list[int] = []
        labels: list[int] = []
        for message in messages:
            role = message["role"].strip().lower()
            prefix = f"<|{role}|>\n"
            content = message["content"].strip()
            segment = f"{prefix}{content}\n"
            segment_ids = self.tokenizer.encode(segment)
            token_ids.extend(segment_ids)
            if role == "assistant":
                prefix_len = len(self.tokenizer.encode(prefix))
                labels.extend([-100] * prefix_len)
                labels.extend(segment_ids[prefix_len:])
            else:
                labels.extend([-100] * len(segment_ids))
        return token_ids, labels

    def _pack_example(self, token_ids: list[int], labels: list[int]) -> tuple[Tensor, Tensor]:
        max_tokens = self.context_length + 1
        if len(token_ids) > max_tokens:
            token_ids = token_ids[-max_tokens:]
            labels = labels[-max_tokens:]
        pad_len = max_tokens - len(token_ids)
        token_ids = token_ids + [0] * pad_len
        labels = labels + [-100] * pad_len
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        indices = torch.randint(0, len(self.examples), (batch_size,))
        x = torch.stack([self.examples[int(i)][0] for i in indices])
        y = torch.stack([self.examples[int(i)][1] for i in indices])
        return x.to(device), y.to(device)
