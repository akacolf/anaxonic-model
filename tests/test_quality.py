import unittest

from anaxonic.quality import (
    quality_failures,
    repeated_character_run,
    repeated_fragment_count,
    unique_word_fraction,
)


class QualityTest(unittest.TestCase):
    def test_repeated_character_run(self):
        self.assertEqual(repeated_character_run("booooom"), 5)

    def test_quality_failures_catch_degenerate_text(self):
        failures = quality_failures(
            "ang ang ang the fond ang ang ang the fond ang ang ang the fond"
        )

        self.assertIn("repeated word sequence", failures)

    def test_quality_failures_catch_generated_fragments(self):
        text = "I ang therele ang aninge wing thelllelere thererereler ang there angeles."

        self.assertGreaterEqual(repeated_fragment_count(text), 6)
        self.assertIn("repeated generated fragments", quality_failures(text))

    def test_quality_allows_short_normal_text(self):
        self.assertEqual(quality_failures("I am ready to help."), [])
        self.assertGreater(unique_word_fraction("I am ready to help."), 0.8)


if __name__ == "__main__":
    unittest.main()
