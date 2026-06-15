import unittest

from anaxonic.train import learning_rate_for_step


class TrainTest(unittest.TestCase):
    def test_learning_rate_warms_up_and_decays(self):
        peak = learning_rate_for_step(10, 100, 1e-3, 1e-4, 10)
        early = learning_rate_for_step(1, 100, 1e-3, 1e-4, 10)
        late = learning_rate_for_step(100, 100, 1e-3, 1e-4, 10)

        self.assertLess(early, peak)
        self.assertEqual(peak, 1e-3)
        self.assertEqual(late, 1e-4)


if __name__ == "__main__":
    unittest.main()
