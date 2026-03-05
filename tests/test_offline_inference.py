# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
End-to-End Offline Inference Tests

Tests the complete vllm-GR inference flow using the refactored API.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@unittest.skipUnless(
    os.environ.get("VLLM_GR_INTEGRATION_TEST"),
    "Requires vLLM + GPU. Set VLLM_GR_INTEGRATION_TEST=1 to run."
)
class TestGRLLMEndToEnd(unittest.TestCase):
    """
    End-to-end tests for GRLLM.

    These tests require a working vLLM installation with GPU.
    Set VLLM_GR_INTEGRATION_TEST=1 to enable.
    """

    def test_single_recommendation(self):
        """Test single user recommendation."""
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

        llm.shutdown()

    def test_batch_recommendation(self):
        """Test batch recommendations."""
        from vllm_gr import GRLLM, GRPrompt
        from typing import List

        llm = GRLLM(model="test-model")

        prompts: List[GRPrompt] = [
            {
                "user_id": f"user_{i}",
                "user_history": [
                    {"item_id": j, "rating": 4, "action_type": 1}
                    for j in range(5)
                ],
                "candidates": [
                    {"item_candidate_id": k * 10}
                    for k in range(10)
                ],
            }
            for i in range(5)
        ]

        outputs = llm.recommend(prompts, use_tqdm=False, top_k=3)

        self.assertEqual(len(outputs), 5)
        for i, output in enumerate(outputs):
            self.assertEqual(output.user_id, f"user_{i}")
            self.assertTrue(output.finished)

        llm.shutdown()


@unittest.skipUnless(
    os.environ.get("VLLM_GR_INTEGRATION_TEST"),
    "Requires vLLM + GPU. Set VLLM_GR_INTEGRATION_TEST=1 to run."
)
class TestGRLLMEngineEndToEnd(unittest.TestCase):
    """
    End-to-end tests for GRLLMEngine.

    These tests require a working vLLM installation with GPU.
    Set VLLM_GR_INTEGRATION_TEST=1 to enable.
    """

    def test_engine_direct_usage(self):
        """Test using GRLLMEngine via from_engine_args."""
        from vllm_gr import GRLLMEngine, GREngineArgs
        from vllm.pooling_params import PoolingParams

        args = GREngineArgs(
            model="test-model",
            max_seq_len=64,
            max_candidates=20,
            default_top_k=3,
        )

        engine = GRLLMEngine.from_engine_args(args)

        prompt = {
            "user_id": "user_001",
            "user_history": [
                {"item_id": 1, "rating": 5, "action_type": 1},
                {"item_id": 2, "rating": 4, "action_type": 2},
            ],
            "candidates": [
                {"item_candidate_id": 10},
                {"item_candidate_id": 20},
                {"item_candidate_id": 30},
            ],
            "num_recommendations": 2,
        }

        engine.add_request(
            request_id="test-001",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )

        self.assertTrue(engine.has_unfinished_requests())

        outputs = []
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            outputs.extend(step_outputs)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].request_id, "test-001")

        engine.shutdown()


class TestConfigModules(unittest.TestCase):
    """Test that config modules can be imported."""

    def test_hstu_config_import(self):
        """Test HSTUConfig import and creation."""
        from vllm_gr.config.hstu_hf_config import HSTUConfig
        config = HSTUConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.hstu_attn_num_layers, 5)
        self.assertEqual(config.hstu_num_heads, 4)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" vllm-GR End-to-End Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestGRLLMEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestGRLLMEngineEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigModules))

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
