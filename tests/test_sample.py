import unittest

from anaxonic.sample import format_chat_prompt, trim_chat_response


class SampleTest(unittest.TestCase):
    def test_format_chat_prompt_uses_training_markers(self):
        self.assertEqual(
            format_chat_prompt("Hello.", "Be brief."),
            "<|system|>\nBe brief.\n<|user|>\nHello.\n<|assistant|>\n",
        )

    def test_trim_chat_response_returns_first_response(self):
        self.assertEqual(trim_chat_response("Yes.\n<|user|>\nAgain?"), "Yes.")
        self.assertEqual(trim_chat_response("Hello.\nextra ramble"), "Hello.")


if __name__ == "__main__":
    unittest.main()
