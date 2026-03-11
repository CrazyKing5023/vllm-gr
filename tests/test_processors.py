# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Input and Output Processors

Tests for:
- GRInputProcessor._extract_features: Feature extraction logic
- GRRequestState._make_gr_output: Output construction with Top-K
- Multi-task output scoring
- GRFeaturesPayload serialization
- GRRequestOutput type hierarchy
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Feature Extraction ────────────────────────────────────────────


class TestExtractFeatures(unittest.TestCase):

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

    def test_extract_missing_key_excluded(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [{"item_id": 1}, {"item_id": 2}]
        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "nonexistent_key"],
        )
        self.assertIn("item_id", keys)
        self.assertNotIn("nonexistent_key", keys)

    def test_extract_empty_records(self):
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        keys, values, lengths = GRInputProcessor._extract_features(
            records=[],
            feature_names=["item_id"],
        )
        self.assertEqual(keys, [])
        self.assertEqual(values, [])
        self.assertEqual(lengths, [])

    def test_extract_default_value(self):
        """Missing values in a record default to 0."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [
            {"item_id": 1, "rating": 5},
            {"item_id": 2},
        ]

        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "rating"],
        )

        rating_idx = keys.index("rating")
        self.assertEqual(values[rating_idx], [5, 0])


# ── Output Construction ──────────────────────────────────────────


class TestGRRequestState(unittest.TestCase):

    def _make_state(self, user_id="user_123", num_recs=3, candidates=None,
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
            gr_user_id=user_id,
            gr_num_recommendations=num_recs,
            gr_original_candidates=candidates,
            gr_primary_id_key=primary_id_key,
        )

    def test_make_gr_output_basic(self):
        """Top-K output with correct ranking order."""
        state = self._make_state(num_recs=3)
        scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])

        from vllm_gr.outputs import GRRequestOutput

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 3)
        self.assertEqual(output.recommendations[0].item_id, 40)
        self.assertEqual(output.recommendations[0].rank, 1)
        self.assertAlmostEqual(output.recommendations[0].score, 0.8)

    def test_make_gr_output_top_k_limit(self):
        """Recommendations limited to num_recommendations."""
        candidates = [{"item_candidate_id": i} for i in range(10)]
        state = self._make_state(num_recs=2, candidates=candidates)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 2)
        self.assertEqual(output.recommendations[0].item_id, 1)
        self.assertEqual(output.recommendations[1].item_id, 3)

    def test_make_gr_output_empty_candidates(self):
        state = self._make_state(num_recs=3, candidates=[])
        scores = torch.tensor([])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 0)
        self.assertTrue(output.finished)

    def test_make_request_output_delegates_for_gr(self):
        """make_request_output intercepts pooling output for GR."""
        state = self._make_state(num_recs=2)
        scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])

        from vllm_gr.outputs import GRRequestOutput
        from vllm.v1.engine import FinishReason

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=scores,
            finish_reason=FinishReason.STOP,
            stop_reason=None,
        )

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(len(output.recommendations), 2)


# ── Multi-Task Output ────────────────────────────────────────────


class TestMultiTaskOutput(unittest.TestCase):

    def _make_state_with_tasks(self, task_names, primary_task="",
                               num_recs=2):
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
            gr_user_id="user_mt",
            gr_num_recommendations=num_recs,
            gr_original_candidates=candidates,
            gr_primary_id_key="item_candidate_id",
            gr_task_names=task_names,
            gr_primary_task=primary_task,
        )

    def test_primary_task_selection(self):
        """Primary task scores are used for ranking."""
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_purchase",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],   # is_click
            [0.2, 0.8, 0.6],   # is_purchase
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(output.recommendations[0].item_id, 20)
        self.assertEqual(output.recommendations[1].item_id, 30)

    def test_per_item_task_scores(self):
        """Each recommended item carries per-task scores."""
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_click",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],
            [0.2, 0.8, 0.6],
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        top = output.recommendations[0]
        self.assertIn("is_click", top.task_scores)
        self.assertIn("is_purchase", top.task_scores)
        self.assertAlmostEqual(top.task_scores["is_click"], 0.9, places=4)

    def test_defaults_to_first_task(self):
        """Without explicit primary_task, first task is used."""
        state = self._make_state_with_tasks(
            task_names=["taskA", "taskB"],
            primary_task="",
        )
        scores = torch.tensor([
            [0.1, 0.9, 0.5],   # taskA
            [0.8, 0.2, 0.6],   # taskB
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(output.recommendations[0].item_id, 20)


# ── GRFeaturesPayload ────────────────────────────────────────────


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
            num_recommendations=5,
        )

        self.assertEqual(payload.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(payload.user_id, "user_123")

        data = msgspec.msgpack.encode(payload)
        decoded = msgspec.msgpack.decode(data, type=GRFeaturesPayload)
        self.assertEqual(decoded.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(decoded.user_id, "user_123")


# ── GRRequestOutput ──────────────────────────────────────────────


class TestGRRequestOutput(unittest.TestCase):

    def test_inherits_pooling_request_output(self):
        from vllm.outputs import PoolingRequestOutput
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        output = GRRequestOutput(
            request_id="test-001",
            user_id="user_1",
            recommendations=[
                RecommendedItem(item_id=10, score=0.9, rank=1),
            ],
            finished=True,
        )
        self.assertIsInstance(output, PoolingRequestOutput)

    def test_properties_and_to_dict(self):
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        output = GRRequestOutput(
            request_id="test-002",
            user_id="user_2",
            recommendations=[
                RecommendedItem(item_id=10, score=0.9, rank=1),
                RecommendedItem(item_id=20, score=0.7, rank=2),
            ],
        )

        self.assertEqual(output.top_recommendation.item_id, 10)
        self.assertEqual(output.item_ids, [10, 20])
        self.assertEqual(output.scores, [0.9, 0.7])

        d = output.to_dict()
        self.assertEqual(d["user_id"], "user_2")
        self.assertEqual(len(d["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
