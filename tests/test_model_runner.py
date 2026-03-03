# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Model and Inference Tests

Tests for HSTU model components and end-to-end inference via EngineCore.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHSTUModelCreation(unittest.TestCase):
    """Tests for HSTUForRecommendation creation and forward pass."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False

    def test_model_creation(self):
        """Test that HSTUForRecommendation can be created from HSTUConfig."""
        if not self.has_torch:
            self.skipTest("torch not available")

        from vllm_gr.config.hstu_hf_config import HSTUConfig
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        config = HSTUConfig(
            hstu_attn_num_layers=2,
            hstu_num_heads=2,
            hstu_transducer_embedding_dim=64,
            hstu_embedding_table_dim=64,
            hstu_attn_linear_dim=64,
            hstu_attn_qk_dim=32,
            hash_size=1000,
            user_embedding_feature_names=["item_id"],
            item_embedding_feature_names=["item_candidate_id"],
            uih_post_id_feature_name="item_id",
            embedding_tables=[
                {
                    "name": "item_embedding",
                    "embedding_dim": 64,
                    "num_embeddings": 1000,
                    "feature_names": ["item_id", "item_candidate_id"],
                },
            ],
        )

        model = HSTUForRecommendation(config=config)
        self.assertIsNotNone(model)

        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)

    def test_dummy_forward(self):
        """Test that forward returns zeros when no KJT features are provided."""
        if not self.has_torch:
            self.skipTest("torch not available")

        from vllm_gr.config.hstu_hf_config import HSTUConfig
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        config = HSTUConfig(
            hstu_attn_num_layers=2,
            hstu_num_heads=2,
            hstu_transducer_embedding_dim=64,
            hstu_embedding_table_dim=64,
            hstu_attn_linear_dim=64,
            hstu_attn_qk_dim=32,
            hash_size=1000,
            embedding_tables=[
                {
                    "name": "item_embedding",
                    "embedding_dim": 64,
                    "num_embeddings": 1000,
                    "feature_names": ["item_id", "item_candidate_id"],
                },
            ],
        )

        model = HSTUForRecommendation(config=config)
        model.eval()

        input_ids = self.torch.tensor([1, 2, 3])
        positions = self.torch.tensor([0, 1, 2])

        with self.torch.no_grad():
            output = model.forward(input_ids, positions)

        self.assertEqual(output.shape, (3, 64))
        self.assertTrue((output == 0).all())


@unittest.skipUnless(
    os.environ.get("VLLM_GR_INTEGRATION_TEST"),
    "Requires vLLM + GPU. Set VLLM_GR_INTEGRATION_TEST=1 to run."
)
class TestEndToEndInference(unittest.TestCase):
    """
    Test end-to-end inference via EngineCore.

    These tests require a working vLLM installation with GPU.
    Set VLLM_GR_INTEGRATION_TEST=1 to enable.
    """

    def setUp(self):
        try:
            import torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False

    def test_grllm_inference(self):
        if not self.has_torch:
            self.skipTest("torch not available")

        from vllm_gr import GRLLM, GRPrompt

        llm = GRLLM(
            model="test-model",
            max_seq_len=64,
            max_candidates=20,
            default_top_k=5,
        )

        prompt: GRPrompt = {
            "user_id": "user_001",
            "user_history": [
                {"item_id": 1, "timestamp": 1000, "rating": 5, "action_type": 1},
                {"item_id": 2, "timestamp": 2000, "rating": 4, "action_type": 1},
                {"item_id": 3, "timestamp": 3000, "rating": 5, "action_type": 2},
            ],
            "candidates": [
                {"item_candidate_id": 10},
                {"item_candidate_id": 20},
                {"item_candidate_id": 30},
                {"item_candidate_id": 40},
                {"item_candidate_id": 50},
            ],
        }

        outputs = llm.recommend([prompt], use_tqdm=False, top_k=3)

        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertEqual(output.user_id, "user_001")
        self.assertTrue(output.finished)
        self.assertLessEqual(len(output.recommendations), 3)

        if len(output.recommendations) > 1:
            scores = [r.score for r in output.recommendations]
            self.assertEqual(scores, sorted(scores, reverse=True))

        llm.shutdown()


def run_all_tests():
    print("=" * 60)
    print(" Model & Inference Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHSTUModelCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndInference))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f" Results: {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
