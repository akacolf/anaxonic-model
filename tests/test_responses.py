import tempfile
import unittest
from pathlib import Path

from anaxonic.responses import ResponseMemory, normalize_prompt, response_is_low_quality


class ResponsesTest(unittest.TestCase):
    def test_normalize_prompt_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_prompt("Hello?!"), "hello")
        self.assertEqual(normalize_prompt("  What's   happening? "), "what s happening")

    def test_response_memory_loads_prompt_response_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat.jsonl"
            path.write_text('{"prompt":"Hello.","response":"Hi there."}\n', encoding="utf-8")

            match = ResponseMemory.from_jsonl(path).lookup("hello")

        self.assertIsNotNone(match)
        self.assertEqual(match.response, "Hi there.")

    def test_response_memory_loads_messages_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat.jsonl"
            path.write_text(
                '{"messages":[{"role":"system","content":"Be brief."},'
                '{"role":"user","content":"Name?"},'
                '{"role":"assistant","content":"Anaxonic."}]}\n',
                encoding="utf-8",
            )

            match = ResponseMemory.from_jsonl(path).lookup("name")

        self.assertIsNotNone(match)
        self.assertEqual(match.response, "Anaxonic.")

    def test_response_memory_rejects_low_confidence_matches(self):
        memory = ResponseMemory([("Hello.", "Hi.")])

        self.assertIsNone(memory.lookup("explain quantum field theory"))

    def test_response_quality_flags_fragmented_generation(self):
        self.assertTrue(response_is_low_quality("Ye ans."))
        self.assertTrue(
            response_is_low_quality(
                "Thes the an ang ting be an the ang and tind the t tiners ang and an"
            )
        )
        self.assertTrue(response_is_low_quality(""))
        self.assertFalse(response_is_low_quality("I am running and ready to help."))


if __name__ == "__main__":
    unittest.main()
