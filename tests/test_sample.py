import unittest

from anaxonic.responses import LOW_CONFIDENCE_RESPONSE
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

    def test_chat_sample_returns_low_confidence_for_unknown_memory_prompt(self):
        self.assertEqual(
            sample(
                "missing-checkpoint.pt",
                "explain tensor parallelism",
                max_new_tokens=20,
                temperature=0.3,
                top_k=20,
                chat=True,
            ),
            LOW_CONFIDENCE_RESPONSE,
        )


if __name__ == "__main__":
    unittest.main()
