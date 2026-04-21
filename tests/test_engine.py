# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Core tests for engine args and GR request data types."""

import os
import sys
import importlib
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_MODEL_DIR = os.path.join(ROOT, "examples", "configs", "movielens-1m")

sys.path.insert(0, ROOT)


def _has_vllm_engine_runtime() -> bool:
    required_modules = [
        "vllm.engine.arg_utils",
        "vllm.platforms",
        "vllm.outputs",
        "vllm.pooling_params",
        "vllm.v1.request",
        "vllm.v1.engine.input_processor",
        "vllm.v1.engine.output_processor",
        "vllm.transformers_utils.model_arch_config_convertor",
        "vllm.entrypoints.llm",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_VLLM_RUNTIME = _has_vllm_engine_runtime()
_SKIP_REASON = (
    "vLLM engine/runtime submodules are unavailable in this environment"
)


# ── Engine Args ───────────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGREngineArgs(unittest.TestCase):

    def test_default_values(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model=DEMO_MODEL_DIR, allow_missing_weights=True)
        self.assertEqual(args.model, DEMO_MODEL_DIR)
        self.assertEqual(args.max_seq_len, 512)
        self.assertEqual(args.max_candidates, 1000)

    def test_custom_values(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            max_seq_len=256,
            max_candidates=500,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
            allow_missing_weights=True,
        )
        self.assertEqual(args.max_seq_len, 256)
        self.assertEqual(args.max_candidates, 500)

    def test_worker_cls_injection(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=False,
        ):
            args = GREngineArgs(
                model=DEMO_MODEL_DIR,
                model_family="hstu",
                allow_missing_weights=True,
            )
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")

    def test_unknown_model_family_does_not_apply_hstu_defaults(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_read_model_type_from_local_config",
            return_value=None,
        ):
            args = GREngineArgs(
                model=DEMO_MODEL_DIR,
                allow_missing_weights=True,
            )
        self.assertNotEqual(
            args.worker_cls,
            "vllm_gr.worker.gr_worker.GRWorker",
        )
        self.assertFalse(args.trust_remote_code)

    def test_grllm_does_not_force_hstu_model_family(self):
        from vllm_gr.entrypoints.gr import GRLLM
        from vllm_gr.engine.arg_utils import GREngineArgs

        captured: dict[str, object] = {}
        fake_engine = mock.Mock()

        def _fake_from_engine_args(*, engine_args, usage_context):
            captured["model_family"] = engine_args.model_family
            return fake_engine

        with mock.patch.object(
            GREngineArgs,
            "_read_model_type_from_local_config",
            return_value=None,
        ), mock.patch(
            "vllm_gr.entrypoints.gr.GRLLMEngine.from_engine_args",
            side_effect=_fake_from_engine_args,
        ):
            llm = GRLLM(
                model=DEMO_MODEL_DIR,
                allow_missing_weights=True,
            )

        self.assertIsNone(captured["model_family"])
        self.assertIsNone(llm.model_family)

    def test_feature_names_not_on_engine_args(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        fields = {f.name for f in GREngineArgs.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)

    def test_dataset_overrides_use_hf_overrides(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            dataset="movielens-1m",
            allow_missing_weights=True,
        )
        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"], "movie_id",
        )

    def test_dataset_overrides_preserve_user_hf_overrides(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            dataset="movielens-1m",
            hf_overrides={"hstu_attn_num_layers": 99},
            allow_missing_weights=True,
        )
        self.assertEqual(args.hf_overrides["hstu_attn_num_layers"], 99)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"], "movie_id",
        )


# ── GRConfig ──────────────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRConfig(unittest.TestCase):

    def test_defaults(self):
        from vllm_gr.config.gr_config import GRConfig

        cfg = GRConfig()
        self.assertEqual(cfg.max_seq_len, 512)
        self.assertEqual(cfg.max_candidates, 1000)

    def test_no_feature_name_fields(self):
        from vllm_gr.config.gr_config import GRConfig

        fields = {f.name for f in GRConfig.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)


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


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
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


if __name__ == "__main__":
    unittest.main()
