import tempfile
import unittest
from pathlib import Path

from anaxonic.build_chat_data import build_chat_dataset, normalize_for_dedupe


class BuildChatDataTest(unittest.TestCase):
    def test_normalize_for_dedupe_ignores_case_spacing_and_punctuation(self):
        self.assertEqual(normalize_for_dedupe(" Hello,   WORLD! "), "hello world")

    def test_build_chat_dataset_filters_duplicates_and_bad_responses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "raw.jsonl"
            out = Path(tmpdir) / "clean.jsonl"
            src.write_text(
                "\n".join(
                    [
                        '{"prompt":"Hi","response":"Hello."}',
                        '{"prompt":"Hi!","response":"Hello"}',
                        '{"prompt":"Browse?","response":"As an AI language model, I cannot browse."}',
                    ]
                ),
                encoding="utf-8",
            )

            written = build_chat_dataset([src], out, seed=1)

            lines = out.read_text(encoding="utf-8").splitlines()

        self.assertEqual(written, 1)
        self.assertEqual(len(lines), 1)
        self.assertIn('"messages"', lines[0])


if __name__ == "__main__":
    unittest.main()
