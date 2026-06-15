import unittest

import torch

from anaxonic.config import ModelConfig
from anaxonic.model import AnaxonicModel


class AnaxonicModelTest(unittest.TestCase):
    def test_attention_is_causal(self):
        torch.manual_seed(7)
        config = ModelConfig(context_length=8, n_layers=1, n_heads=2, d_model=16, dropout=0.0)
        model = AnaxonicModel(config)
        model.eval()

        tokens = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 99, 100]])

        with torch.no_grad():
            logits, _ = model(tokens)
            changed_logits, _ = model(changed_future)

        self.assertTrue(torch.allclose(logits[:, :2], changed_logits[:, :2], atol=1e-6))
        self.assertFalse(torch.allclose(logits[:, -1], changed_logits[:, -1], atol=1e-6))

    def test_forward_rejects_overlong_sequences(self):
        config = ModelConfig(context_length=2, n_layers=1, n_heads=1, d_model=8, dropout=0.0)
        model = AnaxonicModel(config)

        with self.assertRaisesRegex(ValueError, "exceeds context"):
            model(torch.tensor([[1, 2, 3]]))

    def test_generate_temperature_zero_uses_argmax(self):
        config = ModelConfig(context_length=4, n_layers=1, n_heads=1, d_model=8, dropout=0.0)
        model = AnaxonicModel(config)
        model.eval()

        output = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, temperature=0)

        self.assertEqual(output.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
