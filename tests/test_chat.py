import unittest

from anaxonic.chat import build_chat_context


class ChatTest(unittest.TestCase):
    def test_build_chat_context_preserves_history(self):
        self.assertEqual(
            build_chat_context(
                [("user", "Hello."), ("assistant", "Hi."), ("user", "Name?")],
                system="Be brief.",
            ),
            (
                "<|system|>\nBe brief.\n"
                "<|user|>\nHello.\n"
                "<|assistant|>\nHi.\n"
                "<|user|>\nName?\n"
                "<|assistant|>\n"
            ),
        )


if __name__ == "__main__":
    unittest.main()
