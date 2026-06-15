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
        self.assertEqual(normalize_prompt("How are ya doin?"), "how are you doing")
        self.assertEqual(normalize_prompt("wassup"), "what is up")

    def test_response_memory_loads_prompt_response_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat.jsonl"
            path.write_text('{"prompt":"Hello.","response":"Hi there."}\n', encoding="utf-8")

            match = ResponseMemory.from_jsonl(path).lookup("hello")

        self.assertIsNotNone(match)
        self.assertEqual(match.response, "Hi there.")

    def test_response_memory_loads_multiple_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.jsonl"
            second = Path(tmpdir) / "second.jsonl"
            missing = Path(tmpdir) / "missing.jsonl"
            first.write_text('{"prompt":"Hi","response":"Hello."}\n', encoding="utf-8")
            second.write_text('{"prompt":"Bye","response":"Goodbye."}\n', encoding="utf-8")

            memory = ResponseMemory.from_jsonl_files([first, missing, second])

        self.assertEqual(memory.lookup("hi").response, "Hello.")
        self.assertEqual(memory.lookup("bye").response, "Goodbye.")

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

    def test_response_memory_tolerates_casual_filler(self):
        memory = ResponseMemory([("You good?", "Yes. I am good and ready to help.")])

        match = memory.lookup("you good ma boi?")

        self.assertIsNotNone(match)
        self.assertEqual(match.response, "Yes. I am good and ready to help.")

    def test_response_memory_can_match_generic_low_quality_answer(self):
        memory = ResponseMemory([("Explain simply.", "I will use a short and simple answer.")])

        match = memory.lookup("explain photosynthesis simply")

        self.assertIsNotNone(match)
        self.assertTrue(response_is_low_quality(match.response))

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
        self.assertIn("unsure", built_in_response("What should you do if you are unsure?"))
        self.assertEqual(
            built_in_response("what do you know"),
            (
                "I know the local Anaxonic project basics and can answer simple questions "
                "from my training data."
            ),
        )
        self.assertEqual(
            built_in_response("What are you?"),
            "I am Anaxonic, a small local assistant model for the Anaxonic project.",
        )
        self.assertIn("one JSON object per line", built_in_response("What is JSONL?"))
        self.assertIn("fixed questions", built_in_response("Why is a prompt eval set useful?"))
        self.assertIn("duplicate", built_in_response("Why should training examples be deduplicated?"))
        self.assertIn("validation loss", built_in_response("How should I compare two training runs?"))
        self.assertIn("data", built_in_response("Summarize the frontier step in one sentence."))
        self.assertIn("data", built_in_response("Does more training data solve weak answers?"))
        self.assertEqual(
            built_in_response("you good ma boi?"),
            "I am doing well and ready to help.",
        )
        self.assertEqual(
            built_in_response("alright"),
            "Alright. What would you like to work on?",
        )
        self.assertEqual(
            built_in_response("i need some data"),
            "Tell me what topic or format you need, and I will help shape the data.",
        )
        self.assertEqual(
            built_in_response("what is training loss"),
            "Training loss measures how well the model predicts examples it is learning from.",
        )
        self.assertEqual(
            built_in_response("how many tokens are we at"),
            "I cannot see the live token count from this chat prompt.",
        )
        self.assertEqual(
            built_in_response("who created you"),
            "I was built in the local Anaxonic model project.",
        )
        self.assertEqual(
            built_in_response("can you tell me whats the motor's speed"),
            "I need the motor specs or sensor data before I can calculate its speed.",
        )
        self.assertEqual(
            built_in_response("fuck dis shit"),
            "That response missed. Ask again and I will try a clearer reply.",
        )
        self.assertEqual(
            built_in_response("holy cow"),
            "That response missed. Ask again and I will try a clearer reply.",
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
        self.assertTrue(response_is_low_quality("The ale an and ang and ang anerere."))
        self.assertTrue(
            response_is_low_quality(
                "I ang therele ang aninge wing thelllelere thererereler ang there angeles."
            )
        )
        self.assertTrue(response_is_low_quality("I will use a short and simple answer."))
        self.assertTrue(response_is_low_quality("The sentime in the state in the producints."))
        self.assertTrue(response_is_low_quality(""))
        self.assertFalse(response_is_low_quality("I am running and ready to help."))
        self.assertFalse(response_is_low_quality("Plants use sunlight to make food."))


if __name__ == "__main__":
    unittest.main()
