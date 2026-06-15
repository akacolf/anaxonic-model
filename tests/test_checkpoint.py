import unittest

import torch

from anaxonic.checkpoint import normalize_model_state_dict


class CheckpointTest(unittest.TestCase):
    def test_normalize_model_state_dict_removes_legacy_attention_masks(self):
        state = {
            "blocks.0.attn.causal_mask": torch.ones(1),
            "_orig_mod.token_embedding.weight": torch.ones(2, 2),
        }

        normalized = normalize_model_state_dict(state)

        self.assertNotIn("blocks.0.attn.causal_mask", normalized)
        self.assertIn("token_embedding.weight", normalized)


if __name__ == "__main__":
    unittest.main()
