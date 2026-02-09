# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Engine Components

Tests for:
- GREngineArgs: Engine argument configuration
- GRLLMEngine: Core recommendation engine
- GRLLM: High-level API interface
- Module exports
"""

import os
import sys
import unittest
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGREngineArgs(unittest.TestCase):
    """Tests for GREngineArgs."""
    
    def test_default_values(self):
        """Test default argument values."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(model="test-model")
        
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.max_seq_len, 512)
        self.assertEqual(args.max_candidates, 1000)
        self.assertEqual(args.default_top_k, 10)
        self.assertEqual(args.tensor_parallel_size, 1)
        self.assertAlmostEqual(args.gpu_memory_utilization, 0.9)
    
    def test_custom_values(self):
        """Test custom argument values."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(
            model="hstu-large",
            embedding_path="/path/to/embeddings",
            max_seq_len=256,
            max_candidates=500,
            default_top_k=20,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
        )
        
        self.assertEqual(args.model, "hstu-large")
        self.assertEqual(args.embedding_path, "/path/to/embeddings")
        self.assertEqual(args.max_seq_len, 256)
        self.assertEqual(args.max_candidates, 500)
        self.assertEqual(args.default_top_k, 20)
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(
            model="test-model",
            max_seq_len=128,
        )
        
        d = args.to_dict()
        
        self.assertEqual(d["model"], "test-model")
        self.assertEqual(d["max_seq_len"], 128)
    
    def test_validation_max_seq_len(self):
        """Test validation of max_seq_len."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        with self.assertRaises(ValueError):
            GREngineArgs(model="test", max_seq_len=0)
        
        with self.assertRaises(ValueError):
            GREngineArgs(model="test", max_seq_len=-1)
    
    def test_validation_gpu_utilization(self):
        """Test validation of gpu_memory_utilization."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        with self.assertRaises(ValueError):
            GREngineArgs(model="test", gpu_memory_utilization=0)
        
        with self.assertRaises(ValueError):
            GREngineArgs(model="test", gpu_memory_utilization=1.5)


class TestGRLLMEngine(unittest.TestCase):
    """Tests for GRLLMEngine."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        from vllm_gr.engine.gr_engine import GRLLMEngine
        
        engine = GRLLMEngine(
            max_seq_len=128,
            max_candidates=100,
            default_top_k=5,
        )
        
        self.assertEqual(engine.max_seq_len, 128)
        self.assertEqual(engine.max_candidates, 100)
        self.assertEqual(engine.default_top_k, 5)
        self.assertIsNotNone(engine.processor)
        self.assertIsNotNone(engine.output_processor)
    
    def test_engine_from_engine_args(self):
        """Test creating engine from engine args."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        from vllm_gr.engine.gr_engine import GRLLMEngine
        
        args = GREngineArgs(
            model="test-model",
            max_seq_len=64,
            max_candidates=50,
            default_top_k=3,
        )
        
        engine = GRLLMEngine.from_engine_args(args)
        
        self.assertEqual(engine.max_seq_len, 64)
        self.assertEqual(engine.max_candidates, 50)
        self.assertEqual(engine.default_top_k, 3)
    
    def test_add_gr_request(self):
        """Test adding a recommendation request."""
        from vllm_gr.engine.gr_engine import GRLLMEngine
        from vllm_gr.inputs.data import GRPrompt
        
        engine = GRLLMEngine()
        
        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": [
                {"item_id": 1},
                {"item_id": 2},
            ],
            "candidates": [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
            ],
        }
        
        engine.add_gr_request(
            request_id="test-001",
            prompt=prompt,
            num_recommendations=2,
        )
        
        self.assertTrue(engine.has_unfinished_requests())
        self.assertEqual(engine.get_num_unfinished_requests(), 1)
    
    def test_abort_request(self):
        """Test aborting a request."""
        from vllm_gr.engine.gr_engine import GRLLMEngine
        from vllm_gr.inputs.data import GRPrompt
        
        engine = GRLLMEngine()
        
        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": [{"item_id": 1}],
            "candidates": [{"item_id": 10}],
        }
        
        engine.add_gr_request(
            request_id="test-001",
            prompt=prompt,
        )
        
        self.assertTrue(engine.has_unfinished_requests())
        
        engine.abort_request("test-001")
        
        self.assertFalse(engine.has_unfinished_requests())
    
    def test_shutdown(self):
        """Test engine shutdown."""
        from vllm_gr.engine.gr_engine import GRLLMEngine
        
        engine = GRLLMEngine()
        
        # Should not raise
        engine.shutdown()


class TestGRLLM(unittest.TestCase):
    """Tests for GRLLM main interface."""
    
    def test_grllm_initialization(self):
        """Test GRLLM initializes correctly."""
        from vllm_gr.entrypoint.gr import GRLLM
        
        llm = GRLLM(
            model="test-model",
            max_seq_len=64,
            max_candidates=50,
            default_top_k=5,
        )
        
        self.assertEqual(llm.model, "test-model")
        self.assertEqual(llm.max_seq_len, 64)
        self.assertEqual(llm.max_candidates, 50)
        self.assertEqual(llm.default_top_k, 5)
        self.assertIsNotNone(llm.llm_engine)
    
    def test_recommend_single(self):
        """Test single recommendation request."""
        from vllm_gr.entrypoint.gr import GRLLM
        from vllm_gr.inputs.data import GRPrompt
        
        llm = GRLLM(model="test-model")
        
        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": [
                {"item_id": 1, "timestamp": 1000},
                {"item_id": 2, "timestamp": 2000},
            ],
            "candidates": [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
            ],
        }
        
        outputs = llm.recommend(prompt, use_tqdm=False, top_k=2)
        
        # Should return a list (may be empty if no model loaded)
        self.assertIsInstance(outputs, list)
    
    def test_recommend_batch(self):
        """Test batch recommendation requests."""
        from vllm_gr.entrypoint.gr import GRLLM
        from vllm_gr.inputs.data import GRPrompt
        
        llm = GRLLM(model="test-model")
        
        prompts: List[GRPrompt] = [
            {
                "user_id": f"user_{i}",
                "user_history": [{"item_id": j} for j in range(3)],
                "candidates": [{"item_id": j * 10} for j in range(5)],
            }
            for i in range(3)
        ]
        
        outputs = llm.recommend(prompts, use_tqdm=False)
        
        self.assertIsInstance(outputs, list)
    
    def test_recommend_validates_input(self):
        """Test that recommend validates GRPrompt format."""
        from vllm_gr.entrypoint.gr import GRLLM
        
        llm = GRLLM(model="test-model")
        
        # Missing required fields
        invalid_prompt = {
            "user_id": "user_123",
            # Missing user_history and candidates
        }
        
        with self.assertRaises(ValueError):
            llm.recommend(invalid_prompt, use_tqdm=False)
    
    def test_context_manager(self):
        """Test GRLLM as context manager."""
        from vllm_gr.entrypoint.gr import GRLLM
        
        with GRLLM(model="test-model") as llm:
            self.assertIsNotNone(llm.llm_engine)
        
        # After exiting context, should have called shutdown
    
    def test_get_engine(self):
        """Test getting the underlying engine."""
        from vllm_gr.entrypoint.gr import GRLLM
        from vllm_gr.engine.gr_engine import GRLLMEngine
        
        llm = GRLLM(model="test-model")
        
        engine = llm.get_engine()
        
        self.assertIsInstance(engine, GRLLMEngine)


class TestModuleExports(unittest.TestCase):
    """Tests for module-level exports."""
    
    def test_main_package_exports(self):
        """Test that main package exports all required symbols."""
        import vllm_gr
        
        # Data types
        self.assertTrue(hasattr(vllm_gr, 'GRPrompt'))
        self.assertTrue(hasattr(vllm_gr, 'GRInputs'))
        self.assertTrue(hasattr(vllm_gr, 'GRRequestOutput'))
        self.assertTrue(hasattr(vllm_gr, 'RecommendedItem'))
        self.assertTrue(hasattr(vllm_gr, 'GRRequest'))
        
        # Processors
        self.assertTrue(hasattr(vllm_gr, 'GRProcessor'))
        self.assertTrue(hasattr(vllm_gr, 'GROutputProcessor'))
        
        # Engine
        self.assertTrue(hasattr(vllm_gr, 'GREngineArgs'))
        self.assertTrue(hasattr(vllm_gr, 'GRLLMEngine'))
        self.assertTrue(hasattr(vllm_gr, 'GRLLM'))
    
    def test_engine_package_exports(self):
        """Test that engine package exports all required symbols."""
        from vllm_gr import engine
        
        self.assertTrue(hasattr(engine, 'GRProcessor'))
        self.assertTrue(hasattr(engine, 'GROutputProcessor'))
        self.assertTrue(hasattr(engine, 'GREngineArgs'))
        self.assertTrue(hasattr(engine, 'GRLLMEngine'))


if __name__ == "__main__":
    unittest.main()
