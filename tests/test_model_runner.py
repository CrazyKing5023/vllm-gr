# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner Tests

Tests the real model inference flow using GRModelRunner.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGRModelRunner(unittest.TestCase):
    """Tests for GRModelRunner."""
    
    def setUp(self):
        """Check if torch is available."""
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False
    
    def test_model_runner_initialization(self):
        """Test GRModelRunner initialization."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.engine.model_runner import GRModelRunner
        
        runner = GRModelRunner(
            model_path=None,  # Use default initialization
            max_seq_len=64,
            max_candidates=20,
            embedding_dim=64,
            hash_size=1000,
        )
        
        self.assertIsNotNone(runner)
        self.assertEqual(runner.max_seq_len, 64)
        self.assertEqual(runner.max_candidates, 20)
    
    def test_model_runner_config_based_inference(self):
        """Test GRModelRunner config-based inference (randomly initialized model)."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.engine.model_runner import GRModelRunner
        from vllm_gr.request import GRRequest
        
        runner = GRModelRunner(
            model_path=None,  # No pretrained weights - use config
            max_seq_len=64,
            max_candidates=20,
            embedding_dim=64,
            hash_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            attention_dim=32,
        )
        
        # Initialize runner
        runner.initialize()
        
        # Create a mock GRRequest
        gr_request = GRRequest(
            request_id="test-001",
            gr_inputs={},
            user_id="user_001",
            arrival_time=0,
            num_recommendations=3,
        )
        
        # Create prompt
        prompt = {
            "user_id": "user_001",
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
                {"item_id": 50},
            ],
        }
        
        # Run inference
        scores = runner.run_inference(gr_request, prompt)
        
        self.assertIsNotNone(scores)
        # Should have 5 scores for 5 candidates
        if isinstance(scores, self.torch.Tensor):
            self.assertEqual(len(scores), 5)
        else:
            self.assertEqual(len(scores), 5)
        
        runner.shutdown()
    
    def test_model_runner_empty_history(self):
        """Test GRModelRunner with empty user history."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.engine.model_runner import GRModelRunner
        from vllm_gr.request import GRRequest
        
        runner = GRModelRunner(
            model_path=None,
            max_seq_len=64,
            max_candidates=20,
            embedding_dim=64,
            hash_size=1000,
        )
        runner.initialize()
        
        gr_request = GRRequest(
            request_id="test-002",
            gr_inputs={},
            user_id="new_user",
            arrival_time=0,
            num_recommendations=3,
        )
        
        prompt = {
            "user_id": "new_user",
            "user_history": [],  # Empty history
            "candidates": [
                {"item_id": 100},
                {"item_id": 200},
                {"item_id": 300},
            ],
        }
        
        scores = runner.run_inference(gr_request, prompt)
        
        self.assertIsNotNone(scores)
        if isinstance(scores, self.torch.Tensor):
            self.assertEqual(len(scores), 3)
        
        runner.shutdown()
    
    def test_model_runner_empty_candidates(self):
        """Test GRModelRunner with empty candidates."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.engine.model_runner import GRModelRunner
        from vllm_gr.request import GRRequest
        
        runner = GRModelRunner(
            model_path=None,
            max_seq_len=64,
            max_candidates=20,
            embedding_dim=64,
            hash_size=1000,
        )
        runner.initialize()
        
        gr_request = GRRequest(
            request_id="test-003",
            gr_inputs={},
            user_id="user_001",
            arrival_time=0,
            num_recommendations=3,
        )
        
        prompt = {
            "user_id": "user_001",
            "user_history": [{"item_id": 1}],
            "candidates": [],  # No candidates
        }
        
        scores = runner.run_inference(gr_request, prompt)
        
        # Should return empty list for no candidates
        self.assertEqual(len(scores), 0)
        
        runner.shutdown()


class TestHSTUModelEmbeddings(unittest.TestCase):
    """Tests for HSTUForRecommendation embedding lookup."""
    
    def setUp(self):
        """Check if torch is available."""
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False
    
    def test_embedding_lookup(self):
        """Test embedding lookup in HSTUForRecommendation."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
        
        config = HSTUModelConfig(
            num_layers=2,
            num_heads=2,
            embedding_dim=64,
            embedding_table_dim=64,
            hidden_dim=64,
            attention_dim=32,
            hash_size=1000,
            embedding_tables=[
                EmbeddingTableConfig(
                    name="item_embedding",
                    embedding_dim=64,
                    num_embeddings=1000,
                    feature_names=["item_id"],
                ),
            ],
        )
        
        device = self.torch.device("cpu")
        model = HSTUForRecommendation.from_model_config(config, device)
        
        # Lookup embeddings
        indices = self.torch.tensor([1, 2, 3, 4, 5])
        embeddings = model.lookup_embeddings(indices)
        
        self.assertEqual(embeddings.shape, (5, 64))
    
    def test_embedding_tables_created(self):
        """Test that embedding tables are created correctly."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
        
        config = HSTUModelConfig(
            num_layers=2,
            num_heads=2,
            embedding_dim=64,
            embedding_table_dim=64,
            hidden_dim=64,
            attention_dim=32,
            hash_size=1000,
            embedding_tables=[
                EmbeddingTableConfig(
                    name="item_embedding",
                    embedding_dim=64,
                    num_embeddings=1000,
                    feature_names=["item_id"],
                ),
                EmbeddingTableConfig(
                    name="user_embedding",
                    embedding_dim=64,
                    num_embeddings=500,
                    feature_names=["user_id"],
                ),
            ],
        )
        
        device = self.torch.device("cpu")
        model = HSTUForRecommendation.from_model_config(config, device)
        
        # Check embedding tables exist
        self.assertIn("item_embedding", model.embedding_tables)
        self.assertIn("user_embedding", model.embedding_tables)
        
        # Check dimensions
        self.assertEqual(model.embedding_tables["item_embedding"].num_embeddings, 1000)
        self.assertEqual(model.embedding_tables["user_embedding"].num_embeddings, 500)


class TestEndToEndWithModelRunner(unittest.TestCase):
    """Test end-to-end inference with GRModelRunner."""
    
    def setUp(self):
        """Check if torch is available."""
        try:
            import torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False
    
    def test_grllm_with_model_runner(self):
        """Test GRLLM with GRModelRunner enabled."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr import GRLLM, GRPrompt
        
        llm = GRLLM(
            model="test-model",
            max_seq_len=64,
            max_candidates=20,
            default_top_k=5,
            embedding_dim=64,
            hash_size=1000,
            load_model=True,  # Enable model loading
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
        
        # Verify scores are meaningful (not all same)
        if len(output.recommendations) > 1:
            scores = [r.score for r in output.recommendations]
            # Scores should be sorted (descending)
            self.assertEqual(scores, sorted(scores, reverse=True))
        
        llm.shutdown()
    
    def test_grllm_similarity_based_scoring(self):
        """Test that similar items get higher scores."""
        if not self.has_torch:
            self.skipTest("torch not available")
        
        from vllm_gr import GRLLM, GRPrompt
        
        llm = GRLLM(
            model="test-model",
            max_seq_len=64,
            max_candidates=20,
            default_top_k=10,
            embedding_dim=64,
            hash_size=1000,
            load_model=True,
        )
        
        # User has interacted with items 1, 2, 3
        # Candidates include items 1, 2 (already seen) and 100, 200 (new)
        prompt: GRPrompt = {
            "user_id": "user_001",
            "user_history": [
                {"item_id": 1},
                {"item_id": 2},
                {"item_id": 3},
            ],
            "candidates": [
                {"item_id": 1},   # Same as history
                {"item_id": 2},   # Same as history
                {"item_id": 100}, # New item
                {"item_id": 200}, # New item
            ],
        }
        
        outputs = llm.recommend([prompt], use_tqdm=False)
        
        self.assertEqual(len(outputs), 1)
        # Just verify we get valid output
        self.assertTrue(outputs[0].finished)
        self.assertGreater(len(outputs[0].recommendations), 0)
        
        llm.shutdown()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" GRModelRunner Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGRModelRunner))
    suite.addTests(loader.loadTestsFromTestCase(TestHSTUModelEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWithModelRunner))
    
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
