import unittest

from anaxonic.eval_chat import evaluate_cases, score_response


class EvalChatTest(unittest.TestCase):
    def test_score_response_checks_expected_and_forbidden_text(self):
        ok, failures = score_response(
            "Validation loss measures held-out prediction quality.",
            {
                "expected_contains": ["held-out"],
                "expected_not_contains": ["training only"],
                "max_chars": 80,
            },
        )

        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_score_response_reports_failures(self):
        ok, failures = score_response("I made it up.", {"expected_contains": ["unsure"]})

        self.assertFalse(ok)
        self.assertEqual(failures, ["missing: unsure"])

    def test_evaluate_cases_returns_pass_rate(self):
        report = evaluate_cases(
            [{"id": "a", "prompt": "What is 2+2?", "expected_contains": ["4"]}],
            lambda case: "2+2 is 4.",
        )

        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
