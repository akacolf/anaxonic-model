from __future__ import annotations


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer with a fixed 256-token vocabulary."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(int(token_id) % self.vocab_size for token_id in token_ids)
        return data.decode("utf-8", errors="replace")

