import tempfile
import unittest
from pathlib import Path

import torch

from anaxonic.config import ModelConfig
from anaxonic.model import AnaxonicModel
from anaxonic.train import (
    append_metrics,
    autocast_dtype_for_device,
    learning_rate_for_step,
    optimizer_parameter_groups,
)


class TrainTest(unittest.TestCase):
    def test_learning_rate_warms_up_and_decays(self):
        peak = learning_rate_for_step(10, 100, 1e-3, 1e-4, 10)
        early = learning_rate_for_step(1, 100, 1e-3, 1e-4, 10)
        late = learning_rate_for_step(100, 100, 1e-3, 1e-4, 10)

        self.assertLess(early, peak)
        self.assertEqual(peak, 1e-3)
        self.assertEqual(late, 1e-4)

    def test_append_metrics_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            append_metrics(path, {"event": "eval", "step": 1})
            append_metrics(path, {"event": "eval", "step": 2})

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertIn('"step": 1', lines[0])

    def test_optimizer_parameter_groups_skip_decay_for_vectors(self):
        model = AnaxonicModel(ModelConfig(context_length=8, n_layers=1, n_heads=2, d_model=16))

        groups = optimizer_parameter_groups(model, weight_decay=0.1)

        self.assertEqual([group["weight_decay"] for group in groups], [0.1, 0.0])
        decayed = {id(param) for param in groups[0]["params"]}
        not_decayed = {id(param) for param in groups[1]["params"]}
        self.assertFalse(decayed & not_decayed)
        for name, param in model.named_parameters():
            if param.ndim < 2 or name.endswith(".bias"):
                self.assertIn(id(param), not_decayed)
            else:
                self.assertIn(id(param), decayed)

    def test_optimizer_parameter_groups_disable_decay_as_single_group(self):
        model = AnaxonicModel(ModelConfig(context_length=8, n_layers=1, n_heads=2, d_model=16))

        groups = optimizer_parameter_groups(model, weight_decay=0.0)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["weight_decay"], 0.0)
        self.assertEqual(len(groups[0]["params"]), len(list(model.parameters())))

    def test_autocast_dtype_defaults_to_float16_off_cuda(self):
        self.assertIs(autocast_dtype_for_device(torch.device("cpu")), torch.float16)


if __name__ == "__main__":
    unittest.main()
