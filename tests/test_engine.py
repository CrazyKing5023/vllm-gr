# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Engine, Config, Data Types

Tests for:
- GREngineArgs: Engine argument configuration
- GRConfig: Typed config dataclass
- Dataset config factory functions
- Processor class hierarchy
- GRPrompt / GRRequest data types
- Monkey patch mechanism
- Module exports
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Engine Args ───────────────────────────────────────────────────


class TestGREngineArgs(unittest.TestCase):

    def test_default_values(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model="test-model")
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.max_seq_len, 512)
        self.assertEqual(args.max_candidates, 1000)
        self.assertEqual(args.default_top_k, 10)

    def test_custom_values(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model="hstu-large",
            max_seq_len=256,
            max_candidates=500,
            default_top_k=20,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
        )
        self.assertEqual(args.max_seq_len, 256)
        self.assertEqual(args.max_candidates, 500)
        self.assertEqual(args.default_top_k, 20)

    def test_worker_cls_injection(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model="test-model")
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")

    def test_feature_names_not_on_engine_args(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        fields = {f.name for f in GREngineArgs.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)

    def test_dataset_overrides_use_hf_overrides(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model="test-model", dataset="movielens-1m")
        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"], "movie_id",
        )

    def test_dataset_overrides_preserve_user_hf_overrides(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model="test-model",
            dataset="movielens-1m",
            hf_overrides={"hstu_attn_num_layers": 99},
        )
        self.assertEqual(args.hf_overrides["hstu_attn_num_layers"], 99)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"], "movie_id",
        )


# ── GRConfig ──────────────────────────────────────────────────────


class TestGRConfig(unittest.TestCase):

    def test_defaults(self):
        from vllm_gr.config.gr_config import GRConfig

        cfg = GRConfig()
        self.assertEqual(cfg.max_seq_len, 512)
        self.assertEqual(cfg.max_candidates, 1000)
        self.assertEqual(cfg.default_top_k, 10)

    def test_no_feature_name_fields(self):
        from vllm_gr.config.gr_config import GRConfig

        fields = {f.name for f in GRConfig.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)


# ── Dataset Configs ───────────────────────────────────────────────


class TestDatasetConfigs(unittest.TestCase):

    def test_movielens_1m(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("movielens-1m")
        self.assertEqual(cfg["uih_post_id_feature_name"], "movie_id")
        self.assertEqual(cfg["task_names"], ["rating"])

        tables = get_embedding_table_overrides("movielens-1m")
        self.assertEqual(len(tables), 6)

    def test_kuairand_1k(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("kuairand-1k")
        self.assertEqual(cfg["uih_post_id_feature_name"], "video_id")
        self.assertEqual(len(cfg["task_names"]), 8)

        tables = get_embedding_table_overrides("kuairand-1k")
        self.assertEqual(len(tables), 7)

    def test_debug(self):
        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        cfg = get_hstu_config_overrides("debug")
        self.assertEqual(cfg["uih_post_id_feature_name"], "uih_post_id")

        tables = get_embedding_table_overrides("debug")
        self.assertEqual(len(tables), 3)


# ── Processor Hierarchy ──────────────────────────────────────────


class TestProcessorHierarchy(unittest.TestCase):

    def test_input_processor_inherits_vllm(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor
        from vllm.v1.engine.input_processor import InputProcessor

        self.assertTrue(issubclass(GRInputProcessor, InputProcessor))

    def test_output_processor_inherits_vllm(self):
        from vllm_gr.engine.gr_output_processor import GROutputProcessor
        from vllm.v1.engine.output_processor import OutputProcessor

        self.assertTrue(issubclass(GROutputProcessor, OutputProcessor))


# ── Data Types ────────────────────────────────────────────────────


class TestGRPrompt(unittest.TestCase):

    def test_basic_gr_prompt(self):
        from vllm_gr.inputs.data import GRPrompt, is_gr_prompt

        prompt: GRPrompt = {
            "user_id": 123,
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [{"item_id": 10}, {"item_id": 20}],
        }
        self.assertTrue(is_gr_prompt(prompt))

    def test_gr_prompt_with_optional_fields(self):
        from vllm_gr.inputs.data import is_gr_prompt

        prompt = {
            "user_id": "user_abc",
            "user_history": [{"item_id": 1, "timestamp": 1000}],
            "candidates": [{"item_id": 10, "category_id": 1}],
            "num_recommendations": 5,
            "context": {"device_type": 0},
        }
        self.assertTrue(is_gr_prompt(prompt))

    def test_is_gr_prompt_false_cases(self):
        from vllm_gr.inputs.data import is_gr_prompt

        self.assertFalse(is_gr_prompt("not a dict"))
        self.assertFalse(is_gr_prompt(None))
        self.assertFalse(is_gr_prompt({"user_id": 1}))
        self.assertFalse(is_gr_prompt({"user_id": 1, "user_history": []}))
        self.assertFalse(is_gr_prompt({}))


class TestGRRequest(unittest.TestCase):

    def test_without_features(self):
        from vllm_gr.request import GRRequest, is_gr_request
        from vllm.pooling_params import PoolingParams

        request = GRRequest(
            request_id="req1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            eos_token_id=None,
        )
        self.assertIsNone(request.gr_features)
        self.assertFalse(is_gr_request(request))

    def test_with_features(self):
        from vllm_gr.request import GRRequest, is_gr_request
        from vllm_gr.engine import GRFeaturesPayload
        from vllm.pooling_params import PoolingParams

        payload = GRFeaturesPayload(
            uih_keys=["item_id"],
            uih_values_per_key=[[1, 2, 3]],
            uih_lengths_per_key=[3],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[2],
            user_id="user_123",
            num_recommendations=5,
        )
        request = GRRequest(
            gr_features=payload,
            request_id="req2",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            eos_token_id=None,
        )
        self.assertTrue(is_gr_request(request))
        self.assertEqual(request.gr_features.user_id, "user_123")

    def test_is_gr_request_false_for_non_request(self):
        from vllm_gr.request import is_gr_request

        self.assertFalse(is_gr_request(None))
        self.assertFalse(is_gr_request({}))


class TestMonkeyPatch(unittest.TestCase):

    def test_patch_module_imports(self):
        from vllm_gr.patch import (
            apply_patches_if_needed,
            is_patched,
            get_patch_count,
            get_original_request,
        )
        self.assertTrue(callable(apply_patches_if_needed))
        self.assertTrue(callable(is_patched))

    def test_patch_idempotent(self):
        from vllm_gr.patch import apply_patches_if_needed

        count1 = apply_patches_if_needed()
        count2 = apply_patches_if_needed()
        self.assertEqual(count1, count2)


# ── GRLLM API Surface ────────────────────────────────────────────


class TestGRLLM(unittest.TestCase):

    def test_grllm_class_api(self):
        from vllm_gr.entrypoint.gr import GRLLM

        for method in ["recommend", "recommend_batch", "get_engine", "shutdown"]:
            self.assertTrue(hasattr(GRLLM, method), f"Missing: {method}")


# ── Module Exports ────────────────────────────────────────────────


class TestModuleExports(unittest.TestCase):

    def test_main_package_exports(self):
        import vllm_gr

        for name in [
            "GRPrompt", "GRRequestOutput", "RecommendedItem", "GRRequest",
            "GRFeaturesPayload", "GREngineCoreRequest", "GRNewRequestData",
            "GRInputProcessor", "GROutputProcessor",
            "GRWorker", "GRModelRunner",
            "GREngineArgs", "GRLLMEngine", "GRLLM",
            "is_gr_prompt", "is_gr_request", "__version__",
        ]:
            self.assertTrue(hasattr(vllm_gr, name), f"Missing export: {name}")

    def test_engine_package_exports(self):
        from vllm_gr import engine

        self.assertTrue(hasattr(engine, "GRFeaturesPayload"))
        self.assertTrue(hasattr(engine, "GREngineCoreRequest"))


if __name__ == "__main__":
    unittest.main()
