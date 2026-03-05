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
            max_seq_len=256,
            max_candidates=500,
            default_top_k=20,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
        )
        
        self.assertEqual(args.model, "hstu-large")
        self.assertEqual(args.max_seq_len, 256)
        self.assertEqual(args.max_candidates, 500)
        self.assertEqual(args.default_top_k, 20)
    
    def test_worker_cls_injection(self):
        """Test that worker_cls is set to GRWorker for HSTU models."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(model="test-model")
        
        # worker_cls should be injected in __post_init__
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")
    
    def test_feature_names_not_on_engine_args(self):
        """Feature names are read from hf_config, not GREngineArgs."""
        from vllm_gr.engine.arg_utils import GREngineArgs

        fields = {f.name for f in GREngineArgs.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)

    def test_dataset_parameter(self):
        """Test that dataset parameter is accepted and defaults to None."""
        from vllm_gr.engine.arg_utils import GREngineArgs
        
        args = GREngineArgs(model="test-model")
        self.assertIsNone(args.dataset)

        args_ml = GREngineArgs(model="test-model", dataset="movielens-1m")
        self.assertEqual(args_ml.dataset, "movielens-1m")

    def test_dataset_overrides_use_hf_overrides(self):
        """Dataset overrides merge into hf_overrides (not disk mutation)."""
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model="test-model", dataset="movielens-1m")

        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(args.hf_overrides["uih_post_id_feature_name"], "movie_id")
        self.assertIn("embedding_tables", args.hf_overrides)

    def test_dataset_overrides_preserve_user_hf_overrides(self):
        """User-supplied hf_overrides win over dataset defaults."""
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model="test-model",
            dataset="movielens-1m",
            hf_overrides={"hstu_attn_num_layers": 99},
        )

        self.assertEqual(args.hf_overrides["hstu_attn_num_layers"], 99)
        self.assertEqual(args.hf_overrides["uih_post_id_feature_name"], "movie_id")

    def test_no_embedding_path_field(self):
        """embedding_path was removed as dead code."""
        from vllm_gr.engine.arg_utils import GREngineArgs

        self.assertNotIn("embedding_path",
                         {f.name for f in GREngineArgs.__dataclass_fields__.values()})


class TestGRConfig(unittest.TestCase):
    """Tests for the typed GRConfig dataclass."""

    def test_defaults(self):
        from vllm_gr.config.gr_config import GRConfig

        cfg = GRConfig()
        self.assertEqual(cfg.max_seq_len, 512)
        self.assertEqual(cfg.max_candidates, 1000)
        self.assertEqual(cfg.default_top_k, 10)

    def test_custom_values(self):
        from vllm_gr.config.gr_config import GRConfig

        cfg = GRConfig(
            max_seq_len=256,
            max_candidates=500,
            default_top_k=20,
        )
        self.assertEqual(cfg.max_seq_len, 256)
        self.assertEqual(cfg.max_candidates, 500)
        self.assertEqual(cfg.default_top_k, 20)

    def test_no_feature_name_fields(self):
        """Feature names live in hf_config, not GRConfig."""
        from vllm_gr.config.gr_config import GRConfig

        fields = {f.name for f in GRConfig.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)


class TestDatasetConfigs(unittest.TestCase):
    """Tests for dataset_configs factory functions."""

    def test_movielens_1m_overrides(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("movielens-1m")
        self.assertEqual(cfg["uih_post_id_feature_name"], "movie_id")
        self.assertEqual(cfg["uih_action_time_feature_name"], "action_timestamp")
        self.assertEqual(cfg["candidates_querytime_feature_name"], "item_query_time")
        self.assertIn(["movie_id", "item_movie_id"],
                       cfg["merge_uih_candidate_feature_mapping"])
        self.assertEqual(cfg["task_names"], ["rating"])

        tables = get_embedding_table_overrides("movielens-1m")
        table_names = [t["name"] for t in tables]
        self.assertIn("movie_id", table_names)
        self.assertIn("user_id", table_names)
        # movielens-1m has user context embedding tables
        self.assertIn("sex", table_names)
        self.assertEqual(len(tables), 6)

    def test_kuairand_overrides(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("kuairand-1k")
        self.assertEqual(cfg["uih_post_id_feature_name"], "video_id")
        self.assertIn(["video_id", "item_video_id"],
                       cfg["merge_uih_candidate_feature_mapping"])
        self.assertEqual(len(cfg["task_names"]), 8)

        tables = get_embedding_table_overrides("kuairand-1k")
        table_names = [t["name"] for t in tables]
        self.assertIn("video_id", table_names)
        self.assertEqual(len(tables), 7)

    def test_streaming_overrides(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("streaming-400m")
        self.assertEqual(cfg["uih_post_id_feature_name"], "item_id")
        self.assertIn("item_candidate_id", cfg["item_embedding_feature_names"])

        tables = get_embedding_table_overrides("streaming-400m")
        self.assertEqual(len(tables), 3)

    def test_debug_overrides(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("debug")
        self.assertEqual(cfg["uih_post_id_feature_name"], "uih_post_id")
        self.assertEqual(cfg["task_names"], ["vvp100"])

        tables = get_embedding_table_overrides("debug")
        self.assertEqual(len(tables), 3)


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

    def test_candidate_internal_names(self):
        """Candidates use model-internal feature names directly."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [{"item_candidate_id": 10}, {"item_candidate_id": 20}]
        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_candidate_id"],
        )
        self.assertEqual(keys, ["item_candidate_id"])
        self.assertEqual(values[0], [10, 20])

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
        
        # Valid prompt (minimal - item_id only)
        valid_minimal = {
            "user_id": "user_123",
            "user_history": [{"item_id": 1}],
            "candidates": [{"item_id": 10}],
        }
        self.assertTrue(is_gr_prompt(valid_minimal))

        # Valid prompt with rating, action_type, category_id
        valid_full = {
            "user_id": "user_123",
            "user_history": [
                {"item_id": 1, "timestamp": 1000, "rating": 5, "action_type": 1},
            ],
            "candidates": [{"item_id": 10, "category_id": 1}],
        }
        self.assertTrue(is_gr_prompt(valid_full))
        
        # Invalid: missing candidates
        invalid = {
            "user_id": "user_123",
            "user_history": [{"item_id": 1, "rating": 5, "action_type": 1}],
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
