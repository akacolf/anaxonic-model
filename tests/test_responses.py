import tempfile
import unittest
from pathlib import Path

from anaxonic.responses import (
    ResponseMemory,
    built_in_response,
    normalize_prompt,
    response_is_low_quality,
)


class ResponsesTest(unittest.TestCase):
    def test_normalize_prompt_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_prompt("Hello?!"), "hello")
        self.assertEqual(normalize_prompt("  What's   happening? "), "what is happening")

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

    def test_response_memory_does_not_confuse_age_for_name(self):
        memory = ResponseMemory([("What is your name?", "My name is Anaxonic.")])

        self.assertIsNone(memory.lookup("whats your age"))

    def test_builtin_response_answers_common_identity_gaps(self):
        self.assertEqual(
            built_in_response("whats your age"),
            "I do not have an age. I am a local model in the Anaxonic project.",
        )
        self.assertEqual(
            built_in_response("why are you important"),
            "I matter when I give useful answers. Right now I am a small model being improved.",
        )
        self.assertEqual(
            built_in_response("bruhhh"),
            "Yeah, that was not a great answer. Ask me a real question and I will try to answer directly.",
        )

    def test_builtin_response_answers_basic_arithmetic(self):
        self.assertEqual(built_in_response("what is 2+2"), "The answer is 4.")
        self.assertEqual(built_in_response("what is two plus two"), "The answer is 4.")
        self.assertEqual(built_in_response("10 / 4"), "The answer is 5/2.")

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
