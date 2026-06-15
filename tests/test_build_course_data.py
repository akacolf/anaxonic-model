import json
import tempfile
import unittest
from pathlib import Path

from anaxonic.build_course_data import COURSE_EXAMPLES, write_course_dataset


class BuildCourseDataTest(unittest.TestCase):
    def test_write_course_dataset_repeats_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "course.jsonl"

            written = write_course_dataset(output, repeat=2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, len(COURSE_EXAMPLES) * 2)
        self.assertEqual(len(records), len(COURSE_EXAMPLES) * 2)
        self.assertEqual(records[0]["messages"][0]["role"], "user")
        self.assertEqual(records[0]["messages"][1]["role"], "assistant")

    def test_rejects_invalid_repeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_course_dataset(Path(tmpdir) / "course.jsonl", repeat=0)


if __name__ == "__main__":
    unittest.main()
