import unittest

from anaxonic.prepare_hf import record_to_chat, record_to_text


class PrepareHfTest(unittest.TestCase):
    def test_converts_prompt_response_record(self):
        record = {"prompt": "Hello", "response": "Hi there"}

        self.assertEqual(record_to_chat(record), {"prompt": "Hello", "response": "Hi there"})

    def test_converts_alpaca_record_with_input(self):
        record = {"instruction": "Summarize", "input": "A long note", "output": "A short note"}

        self.assertEqual(
            record_to_chat(record),
            {"prompt": "Summarize\n\nA long note", "response": "A short note"},
        )

    def test_converts_sharegpt_conversations(self):
        record = {
            "conversations": [
                {"from": "human", "value": "Name?"},
                {"from": "gpt", "value": "Anaxonic."},
            ]
        }

        self.assertEqual(
            record_to_chat(record),
            {
                "messages": [
                    {"role": "user", "content": "Name?"},
                    {"role": "assistant", "content": "Anaxonic."},
                ]
            },
        )

    def test_reads_text_field(self):
        self.assertEqual(record_to_text({"text": "Plain corpus"}, "text"), "Plain corpus")


if __name__ == "__main__":
    unittest.main()
