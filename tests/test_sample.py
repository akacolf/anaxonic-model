import tempfile
import unittest
from pathlib import Path

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
            "I am ready to help. Please send the question again with a little more detail.",
        )


if __name__ == "__main__":
    unittest.main()
