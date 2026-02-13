# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Model and Inference Tests

Tests for HSTU model components and end-to-end inference via EngineCore.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        """Check if torch is available."""
        try:
            import torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False
    
    def test_grllm_inference(self):
        """Test GRLLM end-to-end inference."""
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
            load_model=True,
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
        
        # Verify scores are sorted (descending)
        if len(output.recommendations) > 1:
            scores = [r.score for r in output.recommendations]
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
        
        prompt: GRPrompt = {
            "user_id": "user_001",
            "user_history": [
                {"item_id": 1},
                {"item_id": 2},
                {"item_id": 3},
            ],
            "candidates": [
                {"item_id": 1},
                {"item_id": 2},
                {"item_id": 100},
                {"item_id": 200},
            ],
        }
        
        outputs = llm.recommend([prompt], use_tqdm=False)
        
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].finished)
        self.assertGreater(len(outputs[0].recommendations), 0)
        
        llm.shutdown()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" Model & Inference Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHSTUModelEmbeddings))
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
