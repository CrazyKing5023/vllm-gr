# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Core tests for GR input/output processor behavior."""

import os
import sys
import importlib
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_vllm_processor_runtime() -> bool:
    required_modules = [
        "vllm.v1.engine",
        "vllm.v1.engine.input_processor",
        "vllm.v1.engine.output_processor",
        "vllm.outputs",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_VLLM_RUNTIME = _has_vllm_processor_runtime()
_SKIP_REASON = (
    "vLLM processor/runtime submodules are unavailable in this environment"
)
_INT32_MAX = (1 << 31) - 1


# ── Feature Extraction ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestExtractFeatures(unittest.TestCase):

    @staticmethod
    def _make_processor_with_context_layout():
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        processor = GRInputProcessor.__new__(GRInputProcessor)
        processor.max_seq_len = 512
        processor.max_candidates = 8
        processor.uih_feature_names = ["item_id", "timestamp"]
        processor.full_uih_feature_names = ["viewer_id", "item_id", "timestamp"]
        processor.context_feature_names = ["viewer_id"]
        processor.candidate_feature_names = ["item_candidate_id", "item_query_time"]
        processor.primary_id_feature_name = "item_id"
        processor._validate_lora = lambda _lora_request: None
        processor._extract_features = GRInputProcessor._extract_features
        processor._extract_context_features = (
            GRInputProcessor._extract_context_features
        )
        processor.attach_schema_fingerprint = (
            GRInputProcessor.attach_schema_fingerprint
        )
        processor.validate_feature_group = GRInputProcessor.validate_feature_group
        processor.validate_context_group = GRInputProcessor.validate_context_group
        processor.vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    model_type="hstu",
                    contextual_feature_to_max_length={"viewer_id": 2},
                    contextual_feature_to_min_uih_length={"viewer_id": 2},
                )
            )
        )
        return processor

    def test_extract_multi_key_features(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [
            {"item_id": 1, "timestamp": 1000, "rating": 4},
            {"item_id": 2, "timestamp": 2000, "rating": 5},
            {"item_id": 3, "timestamp": 3000, "rating": 3},
        ]

        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "timestamp", "rating"],
        )

        self.assertEqual(len(keys), 3)
        item_idx = keys.index("item_id")
        self.assertEqual(values[item_idx], [1, 2, 3])
        for length in lengths:
            self.assertEqual(length, 3)

    def test_extract_missing_key_rejected(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [{"item_id": 1}, {"item_id": 2}]
        with self.assertRaisesRegex(ValueError, "do not match configured schema"):
            GRInputProcessor._extract_features(
                records=records,
                feature_names=["item_id", "nonexistent_key"],
            )

    def test_extract_missing_value_rejected(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [
            {"item_id": 1, "rating": 5},
            {"item_id": 2},
        ]

        with self.assertRaisesRegex(ValueError, "do not match configured schema"):
            GRInputProcessor._extract_features(
                records=records,
                feature_names=["item_id", "rating"],
            )

    def test_process_gr_prompt_preserves_all_candidates(self):
        from vllm.pooling_params import PoolingParams
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        processor = GRInputProcessor.__new__(GRInputProcessor)
        processor.max_seq_len = 512
        processor.max_candidates = 1
        processor.uih_feature_names = ["item_id"]
        processor.full_uih_feature_names = ["item_id"]
        processor.context_feature_names = []
        processor.candidate_feature_names = ["item_candidate_id"]
        processor.primary_id_feature_name = "item_id"
        processor._validate_lora = lambda _lora_request: None
        processor._extract_features = GRInputProcessor._extract_features
        processor._extract_context_features = (
            GRInputProcessor._extract_context_features
        )
        processor.attach_schema_fingerprint = (
            GRInputProcessor.attach_schema_fingerprint
        )
        processor.validate_feature_group = GRInputProcessor.validate_feature_group
        processor.validate_context_group = GRInputProcessor.validate_context_group

        request = processor._process_gr_prompt(
            request_id="req-no-truncation",
            prompt={
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [
                    {"item_candidate_id": 10},
                    {"item_candidate_id": 20},
                    {"item_candidate_id": 30},
                ],
            },
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(
            request.gr_features.cand_values_per_key,
            [[10, 20, 30]],
        )
        self.assertEqual(len(request.gr_features.original_candidates), 3)

    def test_process_gr_prompt_separates_contextual_features_from_history(self):
        from vllm.pooling_params import PoolingParams
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        processor = GRInputProcessor.__new__(GRInputProcessor)
        processor.max_seq_len = 512
        processor.max_candidates = 4
        processor.uih_feature_names = ["uih_post_id", "uih_owner_id"]
        processor.full_uih_feature_names = [
            "viewer_id",
            "uih_post_id",
            "uih_owner_id",
            "dummy_contexual",
        ]
        processor.context_feature_names = ["viewer_id", "dummy_contexual"]
        processor.candidate_feature_names = ["item_post_id"]
        processor.primary_id_feature_name = "uih_post_id"
        processor._validate_lora = lambda _lora_request: None
        processor._extract_features = GRInputProcessor._extract_features
        processor._extract_context_features = (
            GRInputProcessor._extract_context_features
        )
        processor.attach_schema_fingerprint = (
            GRInputProcessor.attach_schema_fingerprint
        )
        processor.validate_feature_group = GRInputProcessor.validate_feature_group
        processor.validate_context_group = GRInputProcessor.validate_context_group

        request = processor._process_gr_prompt(
            request_id="req-contextual-split",
            prompt={
                "user_id": "u1",
                "user_history": [
                    {"uih_post_id": 11, "uih_owner_id": 101},
                    {"uih_post_id": 12, "uih_owner_id": 102},
                ],
                "candidates": [{"item_post_id": 21}],
                "context": {
                    "viewer_id": 7,
                    "dummy_contexual": 1,
                },
            },
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(
            request.gr_features.uih_keys,
            ["uih_post_id", "uih_owner_id"],
        )
        self.assertEqual(
            request.gr_features.ctx_keys,
            ["viewer_id", "dummy_contexual"],
        )
        self.assertEqual(
            request.gr_features.full_uih_keys,
            ["viewer_id", "uih_post_id", "uih_owner_id", "dummy_contexual"],
        )
        self.assertEqual(request.gr_features.uih_values_per_key, [[11, 12], [101, 102]])
        self.assertEqual(request.gr_features.ctx_values_per_key, [[7], [1]])

    def test_process_gr_prompt_builds_stable_prompt_identity(self):
        from vllm.pooling_params import PoolingParams

        processor = self._make_processor_with_context_layout()
        prompt = {
            "user_id": "u-cache",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }

        request_a = processor._process_gr_prompt(
            request_id="req-a",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )
        request_b = processor._process_gr_prompt(
            request_id="req-b",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(request_a.prompt_token_ids, request_b.prompt_token_ids)
        self.assertEqual(request_a.cache_salt, request_b.cache_salt)
        self.assertTrue(request_a.cache_salt.startswith("hstu-prefix-v1:"))
        self.assertEqual(
            len(request_a.prompt_token_ids),
            2 + len(prompt["user_history"]) + len(prompt["candidates"]),
        )
        self.assertTrue(
            all(0 <= token_id <= _INT32_MAX for token_id in request_a.prompt_token_ids)
        )

    def test_process_gr_prompt_prompt_identity_changes_with_context_and_candidates(self):
        from vllm.pooling_params import PoolingParams

        processor = self._make_processor_with_context_layout()
        base_prompt = {
            "user_id": "u-diff",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }
        context_changed_prompt = {
            **base_prompt,
            "context": {"viewer_id": 8},
        }
        candidate_changed_prompt = {
            **base_prompt,
            "candidates": [
                {"item_candidate_id": 99, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
        }

        base_request = processor._process_gr_prompt(
            request_id="req-base",
            prompt=base_prompt,
            params=PoolingParams(task="embed"),
        )
        context_request = processor._process_gr_prompt(
            request_id="req-context",
            prompt=context_changed_prompt,
            params=PoolingParams(task="embed"),
        )
        candidate_request = processor._process_gr_prompt(
            request_id="req-candidate",
            prompt=candidate_changed_prompt,
            params=PoolingParams(task="embed"),
        )

        self.assertNotEqual(
            base_request.prompt_token_ids[:2],
            context_request.prompt_token_ids[:2],
        )
        self.assertEqual(
            base_request.prompt_token_ids[2:5],
            candidate_request.prompt_token_ids[2:5],
        )
        self.assertNotEqual(
            base_request.prompt_token_ids[5:],
            candidate_request.prompt_token_ids[5:],
        )


# ── Output Construction ──────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRRequestState(unittest.TestCase):

    def _make_state(self, user_id="user_123", candidates=None,
                    primary_id_key="item_candidate_id"):
        from vllm_gr.engine.gr_output_processor import GRRequestState

        if candidates is None:
            candidates = [
                {"item_candidate_id": 10},
                {"item_candidate_id": 20},
                {"item_candidate_id": 30},
                {"item_candidate_id": 40},
                {"item_candidate_id": 50},
            ]

        return GRRequestState(
            request_id="test-001",
            external_req_id="test-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_user_id=user_id,
            gr_original_candidates=candidates,
            gr_primary_id_key=primary_id_key,
        )

    def test_make_gr_output_basic(self):
        state = self._make_state()
        scores = torch.tensor([[0.1, 0.5, 0.3, 0.8, 0.2]])

        from vllm_gr.outputs import GRRequestOutput

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 5)
        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.1)

    def test_make_gr_output_preserves_all_candidate_scores(self):
        candidates = [{"item_candidate_id": i} for i in range(10)]
        state = self._make_state(candidates=candidates)
        scores = torch.tensor([[0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15]])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 10)
        self.assertEqual(output.recommendations[0].item_id, 0)
        self.assertEqual(output.recommendations[1].item_id, 1)
        self.assertAlmostEqual(output.recommendations[1].score, 0.9)

    def test_make_gr_output_rejects_candidate_score_mismatch(self):
        state = self._make_state(candidates=[{"item_candidate_id": 10}, {"item_candidate_id": 20}])
        scores = torch.tensor([[0.1]])

        with self.assertRaisesRegex(ValueError, "candidate/score count mismatch"):
            state._make_gr_output(scores, finish_reason="stop")

    def test_make_gr_output_rejects_non_2d_scores(self):
        state = self._make_state()

        with self.assertRaisesRegex(ValueError, "must be 2D"):
            state._make_gr_output(torch.tensor([0.1, 0.5]), finish_reason="stop")

    def test_make_request_output_keeps_empty_candidate_requests_in_gr_path(self):
        state = self._make_state(candidates=[])

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=torch.tensor([0.42]),
            finish_reason="stop",
            stop_reason=None,
        )

        from vllm_gr.outputs import GRRequestOutput

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(output.recommendations, [])

    def test_make_gr_output_uses_explicit_primary_id_key(self):
        state = self._make_state(
            candidates=[
                {"rank_feature": 999, "item_candidate_id": 10},
                {"rank_feature": 555, "item_candidate_id": 20},
            ],
            primary_id_key="item_candidate_id",
        )

        output = state._make_gr_output(
            torch.tensor([[0.1, 0.2]]),
            finish_reason="stop",
        )

        self.assertEqual(output.item_ids, [10, 20])


# ── Multi-Task Output ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestMultiTaskOutput(unittest.TestCase):

    def _make_state_with_tasks(self, task_names, primary_task=""):
        from vllm_gr.engine.gr_output_processor import GRRequestState

        candidates = [
            {"item_candidate_id": 10},
            {"item_candidate_id": 20},
            {"item_candidate_id": 30},
        ]

        return GRRequestState(
            request_id="mt-001",
            external_req_id="mt-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_user_id="user_mt",
            gr_original_candidates=candidates,
            gr_primary_id_key="item_candidate_id",
            gr_task_names=task_names,
            gr_primary_task=primary_task,
        )

    def test_primary_task_selection(self):
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_purchase",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],   # is_click
            [0.2, 0.8, 0.6],   # is_purchase
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.2)
        self.assertEqual(output.recommendations[1].item_id, 20)
        self.assertAlmostEqual(output.recommendations[1].score, 0.8)

    def test_per_item_task_scores(self):
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_click",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],
            [0.2, 0.8, 0.6],
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        first = output.recommendations[0]
        self.assertIn("is_click", first.task_scores)
        self.assertIn("is_purchase", first.task_scores)
        self.assertAlmostEqual(first.task_scores["is_click"], 0.9, places=4)


# ── GRFeaturesPayload ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRFeaturesPayload(unittest.TestCase):

    def test_creation_and_serialization(self):
        """Payload round-trips through msgspec."""
        import msgspec
        from vllm_gr.engine import GRFeaturesPayload

        payload = GRFeaturesPayload(
            uih_keys=["item_id", "timestamp"],
            uih_values_per_key=[[1, 2, 3], [1000, 2000, 3000]],
            uih_lengths_per_key=[3, 3],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[2],
            user_id="user_123",
        )

        self.assertEqual(payload.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(payload.user_id, "user_123")

        data = msgspec.msgpack.encode(payload)
        decoded = msgspec.msgpack.decode(data, type=GRFeaturesPayload)
        self.assertEqual(decoded.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(decoded.user_id, "user_123")


if __name__ == "__main__":
    unittest.main()
