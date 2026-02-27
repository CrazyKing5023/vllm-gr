# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Input and Output Processors

Tests for:
- GRInputProcessor._extract_features: Feature extraction logic
- GRRequestState._make_gr_output: Output construction with Top-K
- GRFeaturesPayload: Feature serialization
"""

import os
import sys
import unittest
from typing import List, Dict, Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExtractFeatures(unittest.TestCase):
    """Tests for GRInputProcessor._extract_features static method."""

    def test_extract_multi_key_features(self):
        """Test extraction with multiple keys."""
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
        self.assertIn("item_id", keys)
        self.assertIn("timestamp", keys)
        self.assertIn("rating", keys)

        item_idx = keys.index("item_id")
        self.assertEqual(values[item_idx], [1, 2, 3])

        ts_idx = keys.index("timestamp")
        self.assertEqual(values[ts_idx], [1000, 2000, 3000])

        rating_idx = keys.index("rating")
        self.assertEqual(values[rating_idx], [4, 5, 3])

        for lens in lengths:
            self.assertEqual(len(lens), 3)
            self.assertTrue(all(l == 1 for l in lens))

    def test_extract_missing_key_excluded(self):
        """Test that missing keys are excluded."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [{"item_id": 1}, {"item_id": 2}]
        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "nonexistent_key"],
        )

        self.assertIn("item_id", keys)
        self.assertNotIn("nonexistent_key", keys)

    def test_extract_empty_records(self):
        """Test extraction with empty records."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        keys, values, lengths = GRInputProcessor._extract_features(
            records=[],
            feature_names=["item_id"],
        )

        self.assertEqual(keys, [])
        self.assertEqual(values, [])
        self.assertEqual(lengths, [])

    def test_extract_default_value(self):
        """Test that missing values default to 0."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        records = [
            {"item_id": 1, "rating": 5},
            {"item_id": 2},  # no rating
        ]

        keys, values, lengths = GRInputProcessor._extract_features(
            records=records,
            feature_names=["item_id", "rating"],
        )

        rating_idx = keys.index("rating")
        self.assertEqual(values[rating_idx], [5, 0])


class TestGRRequestState(unittest.TestCase):
    """Tests for GRRequestState output construction."""

    def _make_state(self, user_id="user_123", num_recs=3, candidates=None):
        from vllm_gr.engine.gr_output_processor import GRRequestState

        if candidates is None:
            candidates = [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
                {"item_id": 40},
                {"item_id": 50},
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
        )

    def test_make_gr_output_basic(self):
        """Test basic GR output creation with Top-K."""
        state = self._make_state(num_recs=3)
        scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])

        from vllm_gr.outputs import GRRequestOutput
        from vllm_gr.engine.gr_output_processor import GRRequestState

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.request_id, "test-001")
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 3)

        # Top 3: item 40 (0.8), item 20 (0.5), item 30 (0.3)
        self.assertEqual(output.recommendations[0].item_id, 40)
        self.assertEqual(output.recommendations[0].rank, 1)
        self.assertAlmostEqual(output.recommendations[0].score, 0.8)

        self.assertEqual(output.recommendations[1].item_id, 20)
        self.assertEqual(output.recommendations[1].rank, 2)

        self.assertEqual(output.recommendations[2].item_id, 30)
        self.assertEqual(output.recommendations[2].rank, 3)

    def test_make_gr_output_top_k_limit(self):
        """Test that recommendations are limited to num_recommendations."""
        candidates = [{"item_id": i} for i in range(10)]
        state = self._make_state(num_recs=2, candidates=candidates)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 2)
        self.assertEqual(output.recommendations[0].item_id, 1)  # score 0.9
        self.assertEqual(output.recommendations[1].item_id, 3)  # score 0.8

    def test_make_gr_output_empty_candidates(self):
        """Test output with empty candidates."""
        state = self._make_state(num_recs=3, candidates=[])
        scores = torch.tensor([])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 0)
        self.assertTrue(output.finished)

    def test_make_request_output_delegates_for_gr(self):
        """Test that make_request_output intercepts GR pooling output."""
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


class TestGRFeaturesPayload(unittest.TestCase):
    """Tests for GRFeaturesPayload serialization."""

    def test_payload_creation(self):
        """Test creating GRFeaturesPayload."""
        from vllm_gr.engine import GRFeaturesPayload

        payload = GRFeaturesPayload(
            uih_keys=["item_id", "timestamp"],
            uih_values_per_key=[[1, 2, 3], [1000, 2000, 3000]],
            uih_lengths_per_key=[[1, 1, 1], [1, 1, 1]],
            cand_keys=["item_id"],
            cand_values_per_key=[[10, 20, 30]],
            cand_lengths_per_key=[[1, 1, 1]],
            user_id="user_123",
            num_recommendations=5,
            original_candidates=[
                {"item_id": 10}, {"item_id": 20}, {"item_id": 30}
            ],
        )

        self.assertEqual(payload.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(payload.uih_values_per_key[0], [1, 2, 3])
        self.assertEqual(payload.user_id, "user_123")
        self.assertEqual(payload.num_recommendations, 5)

    def test_payload_msgspec_serialization(self):
        """Test that GRFeaturesPayload can be serialized/deserialized."""
        import msgspec
        from vllm_gr.engine import GRFeaturesPayload

        payload = GRFeaturesPayload(
            uih_keys=["item_id"],
            uih_values_per_key=[[1, 2, 3]],
            uih_lengths_per_key=[[1, 1, 1]],
            cand_keys=["item_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[[1, 1]],
            user_id="u1",
            num_recommendations=3,
        )

        data = msgspec.msgpack.encode(payload)
        self.assertIsInstance(data, bytes)

        decoded = msgspec.msgpack.decode(data, type=GRFeaturesPayload)
        self.assertEqual(decoded.uih_keys, ["item_id"])
        self.assertEqual(decoded.uih_values_per_key[0], [1, 2, 3])
        self.assertEqual(decoded.user_id, "u1")


class TestGRRequestOutput(unittest.TestCase):
    """Tests for GRRequestOutput (inherits PoolingRequestOutput)."""

    def test_inherits_pooling_request_output(self):
        """Test that GRRequestOutput is a PoolingRequestOutput."""
        from vllm.outputs import PoolingRequestOutput
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        output = GRRequestOutput(
            request_id="test-001",
            user_id="user_1",
            recommendations=[
                RecommendedItem(item_id=10, score=0.9, rank=1),
                RecommendedItem(item_id=20, score=0.8, rank=2),
            ],
            finished=True,
        )

        self.assertIsInstance(output, PoolingRequestOutput)
        self.assertEqual(output.request_id, "test-001")
        self.assertTrue(output.finished)
        self.assertEqual(len(output.recommendations), 2)

    def test_to_dict(self):
        """Test serialization to dict."""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        output = GRRequestOutput(
            request_id="test-002",
            user_id="user_2",
            recommendations=[
                RecommendedItem(item_id=10, score=0.9, rank=1),
            ],
        )

        d = output.to_dict()
        self.assertEqual(d["request_id"], "test-002")
        self.assertEqual(d["user_id"], "user_2")
        self.assertEqual(len(d["recommendations"]), 1)

    def test_properties(self):
        """Test convenience properties."""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        output = GRRequestOutput(
            request_id="test-003",
            user_id="user_3",
            recommendations=[
                RecommendedItem(item_id=10, score=0.9, rank=1),
                RecommendedItem(item_id=20, score=0.7, rank=2),
            ],
        )

        self.assertEqual(output.top_recommendation.item_id, 10)
        self.assertEqual(output.item_ids, [10, 20])
        self.assertEqual(output.scores, [0.9, 0.7])


if __name__ == "__main__":
    unittest.main()
