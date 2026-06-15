import unittest

from anaxonic.release_gate import gate_failures


class ReleaseGateTest(unittest.TestCase):
    def test_gate_passes_when_thresholds_are_met(self):
        report = {
            "pass_rate": 0.9,
            "categories": {"chat": {"pass_rate": 0.8}},
            "results": [{"quality_failures": []}],
        }

        failures = gate_failures(
            report,
            min_pass_rate=0.8,
            min_category_pass_rate=0.75,
            max_quality_failures=0,
        )

        self.assertEqual(failures, [])

    def test_gate_fails_low_scores_and_quality_failures(self):
        report = {
            "pass_rate": 8 / 15,
            "categories": {"raw": {"pass_rate": 8 / 15}},
            "results": [{"quality_failures": ["repeated word sequence"]}],
        }

        failures = gate_failures(
            report,
            min_pass_rate=0.8,
            min_category_pass_rate=0.7,
            max_quality_failures=0,
        )

        self.assertEqual(len(failures), 3)
        self.assertIn("pass_rate 0.533 < required 0.800", failures)


if __name__ == "__main__":
    unittest.main()
