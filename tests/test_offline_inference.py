# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
End-to-End Offline Inference Tests

Tests the complete vllm-GR inference flow using the new API.
"""

import os
import sys
import time
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGRLLMEndToEnd(unittest.TestCase):
    """End-to-end tests for GRLLM."""
    
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
                {"item_id": 1, "timestamp": 1000},
                {"item_id": 2, "timestamp": 2000},
                {"item_id": 3, "timestamp": 3000},
            ],
            "candidates": [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
                {"item_id": 40},
                {"item_id": 50},
            ],
        }
        
        outputs = llm.recommend([prompt], use_tqdm=False, top_k=3)
        
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertEqual(output.user_id, "user_001")
        self.assertTrue(output.finished)
        self.assertLessEqual(len(output.recommendations), 3)
    
    def test_batch_recommendation(self):
        """Test batch recommendations."""
        from vllm_gr import GRLLM, GRPrompt
        from typing import List
        
        llm = GRLLM(model="test-model")
        
        prompts: List[GRPrompt] = [
            {
                "user_id": f"user_{i}",
                "user_history": [{"item_id": j} for j in range(5)],
                "candidates": [{"item_id": k * 10} for k in range(10)],
            }
            for i in range(5)
        ]
        
        outputs = llm.recommend(prompts, use_tqdm=False, top_k=3)
        
        self.assertEqual(len(outputs), 5)
        for i, output in enumerate(outputs):
            self.assertEqual(output.user_id, f"user_{i}")
            self.assertTrue(output.finished)
    
    def test_recommend_batch_method(self):
        """Test recommend_batch with batch_size."""
        from vllm_gr import GRLLM, GRPrompt
        from typing import List
        
        llm = GRLLM(model="test-model")
        
        prompts: List[GRPrompt] = [
            {
                "user_id": f"user_{i}",
                "user_history": [{"item_id": j} for j in range(3)],
                "candidates": [{"item_id": k} for k in range(5)],
            }
            for i in range(10)
        ]
        
        outputs = llm.recommend_batch(prompts, batch_size=3, top_k=2)
        
        self.assertEqual(len(outputs), 10)
    
    def test_output_properties(self):
        """Test GRRequestOutput properties."""
        from vllm_gr import GRLLM, GRPrompt
        
        llm = GRLLM(model="test-model")
        
        prompt: GRPrompt = {
            "user_id": 12345,  # Test with integer user_id
            "user_history": [{"item_id": i} for i in range(10)],
            "candidates": [{"item_id": i * 100} for i in range(20)],
        }
        
        outputs = llm.recommend([prompt], use_tqdm=False, top_k=5)
        output = outputs[0]
        
        # Test properties
        self.assertIsNotNone(output.request_id)
        self.assertEqual(output.user_id, 12345)
        self.assertTrue(output.finished)
        
        # Test top_recommendation property
        if output.recommendations:
            top = output.top_recommendation
            self.assertIsNotNone(top)
            self.assertEqual(top.rank, 1)
        
        # Test item_ids property
        item_ids = output.item_ids
        self.assertIsInstance(item_ids, list)
        
        # Test scores property
        scores = output.scores
        self.assertIsInstance(scores, list)
        
        # Test to_dict
        output_dict = output.to_dict()
        self.assertIn("request_id", output_dict)
        self.assertIn("user_id", output_dict)
        self.assertIn("recommendations", output_dict)
    
    def test_empty_history(self):
        """Test with empty user history."""
        from vllm_gr import GRLLM, GRPrompt
        
        llm = GRLLM(model="test-model")
        
        prompt: GRPrompt = {
            "user_id": "new_user",
            "user_history": [],  # Empty history
            "candidates": [{"item_id": i} for i in range(5)],
        }
        
        outputs = llm.recommend([prompt], use_tqdm=False)
        
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].finished)
    
    def test_context_manager(self):
        """Test GRLLM as context manager."""
        from vllm_gr import GRLLM
        
        with GRLLM(model="test-model") as llm:
            prompt = {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_id": 10}],
            }
            outputs = llm.recommend([prompt], use_tqdm=False)
            self.assertEqual(len(outputs), 1)
    
    def test_invalid_prompt_raises_error(self):
        """Test that invalid prompt raises ValueError."""
        from vllm_gr import GRLLM
        
        llm = GRLLM(model="test-model")
        
        # Missing required fields
        invalid_prompt = {
            "user_id": "u1",
            # Missing user_history and candidates
        }
        
        with self.assertRaises(ValueError):
            llm.recommend([invalid_prompt], use_tqdm=False)


class TestGRLLMEngineEndToEnd(unittest.TestCase):
    """End-to-end tests for GRLLMEngine."""
    
    def test_engine_direct_usage(self):
        """Test using GRLLMEngine directly."""
        from vllm_gr import GRLLMEngine, GREngineArgs
        
        args = GREngineArgs(
            model="test-model",
            max_seq_len=64,
            max_candidates=20,
            default_top_k=3,
        )
        
        engine = GRLLMEngine.from_engine_args(args)
        
        prompt = {
            "user_id": "user_001",
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [{"item_id": 10}, {"item_id": 20}, {"item_id": 30}],
        }
        
        engine.add_gr_request(
            request_id="test-001",
            prompt=prompt,
            num_recommendations=2,
        )
        
        self.assertTrue(engine.has_unfinished_requests())
        
        outputs = []
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            outputs.extend(step_outputs)
        
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].request_id, "test-001")
        
        engine.shutdown()
    
    def test_engine_multiple_requests(self):
        """Test engine with multiple requests."""
        from vllm_gr import GRLLMEngine
        
        engine = GRLLMEngine(
            max_seq_len=32,
            max_candidates=10,
            default_top_k=2,
        )
        
        for i in range(3):
            prompt = {
                "user_id": f"user_{i}",
                "user_history": [{"item_id": j} for j in range(3)],
                "candidates": [{"item_id": k * 10} for k in range(5)],
            }
            engine.add_gr_request(
                request_id=f"req-{i}",
                prompt=prompt,
            )
        
        self.assertEqual(engine.get_num_unfinished_requests(), 3)
        
        outputs = []
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            outputs.extend(step_outputs)
        
        self.assertEqual(len(outputs), 3)
        
        engine.shutdown()


class TestProcessorsEndToEnd(unittest.TestCase):
    """End-to-end tests for processors."""
    
    def test_processor_full_flow(self):
        """Test GRProcessor and GROutputProcessor together."""
        from vllm_gr import GRProcessor, GROutputProcessor, GRPrompt
        
        # Create processors
        processor = GRProcessor(max_seq_len=10, max_candidates=5)
        output_processor = GROutputProcessor(default_top_k=3)
        
        # Create prompt
        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": [
                {"item_id": 1},
                {"item_id": 2},
                {"item_id": 3},
            ],
            "candidates": [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
                {"item_id": 40},
            ],
        }
        
        # Process input
        gr_request = processor.process_gr_inputs(
            request_id="test-001",
            prompt=prompt,
            num_recommendations=3,
        )
        
        self.assertEqual(gr_request.request_id, "test-001")
        self.assertEqual(gr_request.user_id, "user_123")
        self.assertEqual(gr_request.num_recommendations, 3)
        
        # Add to output processor
        output_processor.add_gr_request(
            gr_request,
            original_candidates=prompt["candidates"],
        )
        
        self.assertTrue(output_processor.has_unfinished_requests())
        
        # Create mock scores and process output
        scores = [0.2, 0.8, 0.5, 0.3]  # item 20 has highest score
        output = output_processor.create_output_from_scores(
            request_id="test-001",
            scores=scores,
        )
        
        self.assertIsNotNone(output)
        self.assertEqual(output.request_id, "test-001")
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 3)
        
        # Verify ranking
        self.assertEqual(output.recommendations[0].item_id, 20)  # Highest score
        self.assertEqual(output.recommendations[0].rank, 1)
        
        # Should have removed the request
        self.assertFalse(output_processor.has_unfinished_requests())


class TestConfigModules(unittest.TestCase):
    """Test that config modules can be imported."""
    
    def test_hstu_config_import(self):
        """Test HSTUModelConfig import and creation."""
        try:
            from vllm_gr.config.hstu_config import HSTUModelConfig
            config = HSTUModelConfig()
            self.assertIsNotNone(config)
            self.assertEqual(config.num_layers, 5)
            self.assertEqual(config.num_heads, 4)
        except ImportError:
            self.skipTest("Config module not available")
    


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" vllm-GR End-to-End Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGRLLMEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestGRLLMEngineEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessorsEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigModules))
    
    # Run tests
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
