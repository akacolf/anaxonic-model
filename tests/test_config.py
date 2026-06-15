import unittest

from anaxonic.config import ModelConfig


class ModelConfigTest(unittest.TestCase):
    def test_model_config_validates_head_size(self):
        config = ModelConfig(d_model=128, n_heads=8)
        self.assertEqual(config.d_head, 16)

    def test_model_config_rejects_bad_head_size(self):
        config = ModelConfig(d_model=130, n_heads=8)
        with self.assertRaisesRegex(ValueError, "d_model"):
            config.d_head


if __name__ == "__main__":
    unittest.main()
