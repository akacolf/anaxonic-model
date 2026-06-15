from __future__ import annotations

from pathlib import Path

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

