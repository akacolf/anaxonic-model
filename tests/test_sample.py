import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from anaxonic.tokenizer import ByteTokenizer
from anaxonic.sample import format_chat_prompt, trim_chat_response
from anaxonic.sample import sample


class SampleTest(unittest.TestCase):
    def test_format_chat_prompt_uses_training_markers(self):
        self.assertEqual(
            format_chat_prompt("Hello.", "Be brief."),
            "<|system|>\nBe brief.\n<|user|>\nHello.\n<|assistant|>\n",
        )

    def test_trim_chat_response_returns_first_response(self):
        self.assertEqual(trim_chat_response("Yes.\n<|user|>\nAgain?"), "Yes.")
        self.assertEqual(trim_chat_response("Hello.\nextra ramble"), "Hello.")

    def test_chat_sample_can_disable_generation_for_unknown_memory_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.jsonl"
            memory_path.write_text('{"prompt":"Hello.","response":"Hi."}\n', encoding="utf-8")

            response = sample(
                "missing-checkpoint.pt",
                "explain tensor parallelism",
                max_new_tokens=20,
                temperature=0.3,
                top_k=20,
                chat=True,
                memory_data=str(memory_path),
                generate_unknown=False,
            )

        self.assertEqual(
            response,
            "I do not know that yet. Ask a clearer question or add a training example for it.",
        )

    def test_chat_sample_uses_builtin_response_before_checkpoint(self):
        response = sample(
            "missing-checkpoint.pt",
            "what is 2+2",
            max_new_tokens=20,
            temperature=0.3,
            top_k=20,
            chat=True,
            memory_data=None,
            generate_unknown=False,
        )

        self.assertEqual(response, "The answer is 4.")

    def test_raw_generation_bypasses_routing_and_quality_fallback(self):
        tokenizer = ByteTokenizer()

        class FakeModel:
            def __init__(self, config):
                self.config = config

            def to(self, device):
                return self

            def eval(self):
                return None

            def generate(self, input_ids, *args, **kwargs):
                response_ids = tokenizer.encode("x\n")
                return torch.tensor([input_ids[0].tolist() + response_ids], dtype=torch.long)

        checkpoint = {
            "config": {
                "model": {
                    "vocab_size": 256,
                    "context_length": 256,
                    "n_layers": 1,
                    "n_heads": 1,
                    "d_model": 16,
                    "dropout": 0.0,
                }
            },
            "model": {},
        }

        with (
            patch("anaxonic.sample.choose_device", return_value=torch.device("cpu")),
            patch("anaxonic.sample.load_checkpoint", return_value=checkpoint),
            patch("anaxonic.sample.load_model_state", return_value=None),
            patch("anaxonic.sample.AnaxonicModel", FakeModel),
        ):
            response = sample(
                "checkpoint.pt",
                "what is 2+2",
                max_new_tokens=20,
                temperature=0,
                top_k=None,
                chat=True,
                memory_data=None,
                use_routing=False,
            )

        self.assertEqual(response, "x")


if __name__ == "__main__":
    unittest.main()
