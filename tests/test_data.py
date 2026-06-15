import tempfile
import unittest
from pathlib import Path

import torch

from anaxonic.data import ConversationDataset
from anaxonic.tokenizer import ByteTokenizer


class ConversationDatasetTest(unittest.TestCase):
    def test_prompt_response_records_mask_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat.jsonl"
            path.write_text('{"prompt":"Hi","response":"Hello"}\n', encoding="utf-8")

            tokenizer = ByteTokenizer()
            dataset = ConversationDataset(path, tokenizer, context_length=64)
            x, y = dataset.sample_batch(batch_size=1, device=torch.device("cpu"))

            decoded_input = tokenizer.decode(x[0].tolist())
            supervised = [token for token in y[0].tolist() if token != -100]

        self.assertIn("<|user|>\nHi", decoded_input)
        self.assertIn("<|assistant|>", decoded_input)
        self.assertEqual(tokenizer.decode(supervised), "Hello\n")

    def test_messages_records_are_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat.jsonl"
            path.write_text(
                '{"messages":[{"role":"user","content":"Name?"},'
                '{"role":"assistant","content":"Anaxonic."}]}\n',
                encoding="utf-8",
            )

            dataset = ConversationDataset(path, ByteTokenizer(), context_length=64)
            x, y = dataset.sample_batch(batch_size=2, device=torch.device("cpu"))

        self.assertEqual(tuple(x.shape), (2, 64))
        self.assertEqual(tuple(y.shape), (2, 64))
        self.assertTrue(torch.any(y != -100))


if __name__ == "__main__":
    unittest.main()
