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
import importlib
import importlib.util
import types
import unittest
from unittest import mock

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

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=False,
        ):
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


class TestInitializationBoundaries(unittest.TestCase):

    def test_main_package_import_is_lightweight(self):
        module_names = [
            "vllm_gr",
            "vllm_gr.patch",
            "vllm_gr.plugin",
            "vllm_gr.config.hstu_hf_config",
            "vllm_gr.model_executor.models.registry",
        ]
        saved_modules = {
            name: sys.modules.get(name)
            for name in module_names
        }

        for name in module_names:
            sys.modules.pop(name, None)

        try:
            module = importlib.import_module("vllm_gr")
            self.assertEqual(module.__version__, "0.3.0")
            self.assertNotIn("vllm_gr.patch", sys.modules)
            self.assertNotIn("vllm_gr.plugin", sys.modules)
            self.assertNotIn("vllm_gr.config.hstu_hf_config", sys.modules)
            self.assertNotIn(
                "vllm_gr.model_executor.models.registry",
                sys.modules,
            )
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
            for name, module in saved_modules.items():
                if module is not None:
                    sys.modules[name] = module

    def test_plugin_initialize_runtime_calls_expected_steps(self):
        from vllm_gr.plugin import initialize_runtime

        fake_patch = types.ModuleType("vllm_gr.patch")
        fake_patch.re_apply_patches = mock.Mock(return_value=3)
        fake_cfg = types.ModuleType("vllm_gr.config.hstu_hf_config")
        fake_cfg.ensure_hstu_config_registered = mock.Mock(return_value=True)
        fake_registry = types.ModuleType("vllm_gr.model_executor.models.registry")
        fake_registry.register_gr_models_to_vllm = mock.Mock(return_value=True)

        fake_config_pkg = types.ModuleType("vllm_gr.config")
        fake_executor_pkg = types.ModuleType("vllm_gr.model_executor")
        fake_models_pkg = types.ModuleType("vllm_gr.model_executor.models")

        with mock.patch.dict(
            sys.modules,
            {
                "vllm_gr.patch": fake_patch,
                "vllm_gr.config": fake_config_pkg,
                "vllm_gr.config.hstu_hf_config": fake_cfg,
                "vllm_gr.model_executor": fake_executor_pkg,
                "vllm_gr.model_executor.models": fake_models_pkg,
                "vllm_gr.model_executor.models.registry": fake_registry,
            },
        ):
            state = initialize_runtime()

        fake_patch.re_apply_patches.assert_called_once_with()
        fake_cfg.ensure_hstu_config_registered.assert_called_once_with()
        fake_registry.register_gr_models_to_vllm.assert_called_once_with()
        self.assertEqual(
            state,
            {
                "patch_count": 3,
                "config_registered": True,
                "models_registered": True,
            },
        )

    @unittest.skipUnless(
        importlib.util.find_spec("vllm") is not None,
        "vllm is required for GREngineArgs integration tests",
    )
    def test_create_engine_config_initializes_runtime(self):
        from vllm.engine.arg_utils import EngineArgs
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model="test-model")

        with mock.patch(
            "vllm_gr.plugin.initialize_runtime",
        ) as initialize_runtime, mock.patch.object(
            EngineArgs,
            "create_engine_config",
            autospec=True,
            return_value=mock.sentinel.engine_config,
        ) as super_create_engine_config:
            result = args.create_engine_config()

        initialize_runtime.assert_called_once_with()
        super_create_engine_config.assert_called_once_with(args, None)
        self.assertIs(result, mock.sentinel.engine_config)


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
