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
    
    def test_to_gr_dict(self):
        """Test converting GR-specific params to dictionary."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(
            model="test-model",
            max_seq_len=128,
        )
        
        d = args.to_gr_dict()
        
        self.assertEqual(d["max_seq_len"], 128)
        self.assertIn("embedding_dim", d)
        self.assertIn("hash_size", d)
    
    def test_worker_cls_injection(self):
        """Test that worker_cls is set to GRWorker for HSTU models."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(model="test-model")
        
        # worker_cls should be injected in __post_init__
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")
    
    def test_feature_names_defaults(self):
        """Test that feature name defaults are correct."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(model="test-model")
        
        # Should include multi-key feature names by default
        self.assertIn("item_id", args.uih_feature_names)
        self.assertIn("item_id", args.candidate_feature_names)


class TestGRInputProcessor(unittest.TestCase):
    """Tests for GRInputProcessor (unit tests for static methods)."""

    def test_extract_features_basic(self):
        """Test _extract_features static method."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [
            {"item_id": 1, "timestamp": 1000},
            {"item_id": 2, "timestamp": 2000},
        ]

        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "timestamp"],
        )

        self.assertIn("item_id", keys)
        self.assertIn("timestamp", keys)
        item_idx = keys.index("item_id")
        self.assertEqual(values[item_idx], [1, 2])

    def test_output_processor_inherits_vllm(self):
        """Test GROutputProcessor inherits from vLLM OutputProcessor."""
        from vllm_gr.engine.gr_output_processor import GROutputProcessor
        from vllm.v1.engine.output_processor import OutputProcessor

        self.assertTrue(issubclass(GROutputProcessor, OutputProcessor))

    def test_input_processor_inherits_vllm(self):
        """Test GRInputProcessor inherits from vLLM InputProcessor."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor
        from vllm.v1.engine.input_processor import InputProcessor

        self.assertTrue(issubclass(GRInputProcessor, InputProcessor))


class TestGRLLM(unittest.TestCase):
    """
    Tests for GRLLM main interface.

    Note: GRLLM.__init__() calls GRLLMEngine.from_engine_args() which
    requires a working vLLM installation with GPU. Tests that create
    a GRLLM instance are marked with @unittest.skip and should be run
    in an environment with vLLM installed and a GPU available.
    """
    
    def test_grllm_import(self):
        """Test GRLLM can be imported."""
        from vllm_gr.entrypoint.gr import GRLLM
        self.assertIsNotNone(GRLLM)
    
    def test_grllm_class_has_recommend(self):
        """Test GRLLM has recommend method."""
        from vllm_gr.entrypoint.gr import GRLLM
        self.assertTrue(hasattr(GRLLM, 'recommend'))
        self.assertTrue(hasattr(GRLLM, 'recommend_batch'))
        self.assertTrue(hasattr(GRLLM, 'get_engine'))
        self.assertTrue(hasattr(GRLLM, 'shutdown'))
    
    def test_gr_prompt_validation_logic(self):
        """Test GRPrompt validation logic (used by recommend)."""
        from vllm_gr.inputs.data import is_gr_prompt
        
        # Valid prompt
        valid = {
            "user_id": "user_123",
            "user_history": [{"item_id": 1}],
            "candidates": [{"item_id": 10}],
        }
        self.assertTrue(is_gr_prompt(valid))
        
        # Invalid: missing candidates
        invalid = {
            "user_id": "user_123",
            "user_history": [{"item_id": 1}],
        }
        self.assertFalse(is_gr_prompt(invalid))


class TestModuleExports(unittest.TestCase):
    """Tests for module-level exports."""
    
    def test_main_package_exports(self):
        """Test that main package exports all required symbols."""
        import vllm_gr
        
        # Data types
        self.assertTrue(hasattr(vllm_gr, 'GRPrompt'))
        self.assertTrue(hasattr(vllm_gr, 'GRRequestOutput'))
        self.assertTrue(hasattr(vllm_gr, 'RecommendedItem'))
        self.assertTrue(hasattr(vllm_gr, 'GRRequest'))
        
        # Pipeline types (new)
        self.assertTrue(hasattr(vllm_gr, 'GRFeaturesPayload'))
        self.assertTrue(hasattr(vllm_gr, 'GREngineCoreRequest'))
        self.assertTrue(hasattr(vllm_gr, 'GRNewRequestData'))
        
        # Processors (updated: GRInputProcessor replaces GRProcessor)
        self.assertTrue(hasattr(vllm_gr, 'GRInputProcessor'))
        self.assertTrue(hasattr(vllm_gr, 'GROutputProcessor'))
        
        # Worker components (new)
        self.assertTrue(hasattr(vllm_gr, 'GRWorker'))
        self.assertTrue(hasattr(vllm_gr, 'GRModelRunner'))
        
        # Engine
        self.assertTrue(hasattr(vllm_gr, 'GREngineArgs'))
        self.assertTrue(hasattr(vllm_gr, 'GRLLMEngine'))
        self.assertTrue(hasattr(vllm_gr, 'GRLLM'))
    
    def test_engine_package_exports(self):
        """Test that engine package exports pipeline types."""
        from vllm_gr import engine
        
        self.assertTrue(hasattr(engine, 'GRFeaturesPayload'))
        self.assertTrue(hasattr(engine, 'GREngineCoreRequest'))


if __name__ == "__main__":
    unittest.main()
