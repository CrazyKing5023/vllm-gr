# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Input and Output Processors

Tests for:
- GRInputProcessor: Input processor (GRPrompt -> GREngineCoreRequest)
- GROutputProcessor: Output processor (engine outputs -> GRRequestOutput)
- GRFeaturesPayload: Feature serialization
"""

import os
import sys
import unittest
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGRInputProcessor(unittest.TestCase):
    """Tests for GRInputProcessor input processor."""

    def setUp(self):
        """Set up test fixtures."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        self.processor = GRInputProcessor(
            max_seq_len=100,
            max_candidates=50,
        )

        # Sample user history (with rich features)
        self.user_history = [
            {"item_id": 1, "timestamp": 1000, "rating": 4},
            {"item_id": 2, "timestamp": 2000, "rating": 5},
            {"item_id": 3, "timestamp": 3000, "rating": 3},
        ]

        # Sample candidates
        self.candidates = [
            {"item_id": 10, "category_id": 1},
            {"item_id": 20, "category_id": 2},
            {"item_id": 30, "category_id": 1},
            {"item_id": 40, "category_id": 3},
            {"item_id": 50, "category_id": 2},
        ]

    def test_processor_initialization(self):
        """Test processor initializes with correct parameters."""
        self.assertEqual(self.processor.max_seq_len, 100)
        self.assertEqual(self.processor.max_candidates, 50)
        # Default feature names include multi-key features
        self.assertIn("item_id", self.processor.uih_feature_names)
        self.assertIn("timestamp", self.processor.uih_feature_names)
        self.assertIn("item_id", self.processor.candidate_feature_names)

    def test_process_gr_prompt_basic(self):
        """Test basic input processing creates GREngineCoreRequest."""
        from vllm_gr.inputs.data import GRPrompt

        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": self.user_history,
            "candidates": self.candidates,
        }

        request = self.processor.process_gr_prompt(
            request_id="test-001",
            prompt=prompt,
            num_recommendations=3,
        )

        # Verify request has gr_features
        self.assertEqual(request.request_id, "test-001")
        self.assertIsNotNone(request.gr_features)
        self.assertEqual(request.gr_features.user_id, "user_123")
        self.assertEqual(request.gr_features.num_recommendations, 3)

    def test_gr_features_payload_contains_all_features(self):
        """Test that ALL features (item_id, timestamp, rating) are serialized."""
        from vllm_gr.inputs.data import GRPrompt

        prompt: GRPrompt = {
            "user_id": 456,
            "user_history": self.user_history,
            "candidates": self.candidates,
        }

        request = self.processor.process_gr_prompt(
            request_id="test-002",
            prompt=prompt,
            num_recommendations=5,
        )

        payload = request.gr_features
        self.assertIsNotNone(payload)

        # Check UIH keys include multi-key features
        self.assertIn("item_id", payload.uih_keys)
        self.assertIn("timestamp", payload.uih_keys)
        self.assertIn("rating", payload.uih_keys)

        # Check UIH values
        item_id_idx = payload.uih_keys.index("item_id")
        self.assertEqual(payload.uih_values_per_key[item_id_idx], [1, 2, 3])

        timestamp_idx = payload.uih_keys.index("timestamp")
        self.assertEqual(
            payload.uih_values_per_key[timestamp_idx], [1000, 2000, 3000]
        )

        rating_idx = payload.uih_keys.index("rating")
        self.assertEqual(
            payload.uih_values_per_key[rating_idx], [4, 5, 3]
        )

        # Check candidate keys
        self.assertIn("item_id", payload.cand_keys)
        self.assertIn("category_id", payload.cand_keys)

        cand_item_idx = payload.cand_keys.index("item_id")
        self.assertEqual(
            payload.cand_values_per_key[cand_item_idx],
            [10, 20, 30, 40, 50],
        )

    def test_prompt_token_ids_are_item_ids(self):
        """Test that prompt_token_ids contains item_ids for vLLM compatibility."""
        from vllm_gr.inputs.data import GRPrompt

        prompt: GRPrompt = {
            "user_id": "u1",
            "user_history": self.user_history,
            "candidates": self.candidates,
        }

        request = self.processor.process_gr_prompt(
            request_id="test-003",
            prompt=prompt,
            num_recommendations=3,
        )

        # prompt_token_ids should be the UIH item_ids
        self.assertEqual(request.prompt_token_ids, [1, 2, 3])

    def test_original_candidates_preserved(self):
        """Test that original candidates are preserved for output mapping."""
        from vllm_gr.inputs.data import GRPrompt

        prompt: GRPrompt = {
            "user_id": "u1",
            "user_history": self.user_history,
            "candidates": self.candidates,
        }

        request = self.processor.process_gr_prompt(
            request_id="test-004",
            prompt=prompt,
            num_recommendations=3,
        )

        self.assertEqual(len(request.gr_features.original_candidates), 5)
        self.assertEqual(
            request.gr_features.original_candidates[0]["item_id"], 10
        )

    def test_max_seq_len_truncation(self):
        """Test that long sequences are truncated to max_seq_len."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor
        from vllm_gr.inputs.data import GRPrompt

        processor = GRInputProcessor(max_seq_len=2, max_candidates=50)

        long_history = [
            {"item_id": 1, "timestamp": 1000},
            {"item_id": 2, "timestamp": 2000},
            {"item_id": 3, "timestamp": 3000},
            {"item_id": 4, "timestamp": 4000},
        ]

        prompt: GRPrompt = {
            "user_id": "u1",
            "user_history": long_history,
            "candidates": self.candidates,
        }

        request = processor.process_gr_prompt(
            request_id="test-005",
            prompt=prompt,
            num_recommendations=3,
        )

        # Only last 2 items should be kept
        self.assertEqual(request.prompt_token_ids, [3, 4])

    def test_max_candidates_truncation(self):
        """Test that candidates are truncated to max_candidates."""
        from vllm_gr.engine.gr_input_processor import GRInputProcessor
        from vllm_gr.inputs.data import GRPrompt

        processor = GRInputProcessor(max_seq_len=100, max_candidates=2)

        prompt: GRPrompt = {
            "user_id": "u1",
            "user_history": self.user_history,
            "candidates": self.candidates,
        }

        request = processor.process_gr_prompt(
            request_id="test-006",
            prompt=prompt,
            num_recommendations=2,
        )

        # Only first 2 candidates should be kept
        self.assertEqual(len(request.gr_features.original_candidates), 2)

    def test_empty_history(self):
        """Test handling of empty user history."""
        from vllm_gr.inputs.data import GRPrompt

        prompt: GRPrompt = {
            "user_id": "u1",
            "user_history": [],
            "candidates": self.candidates,
        }

        request = self.processor.process_gr_prompt(
            request_id="test-007",
            prompt=prompt,
            num_recommendations=3,
        )

        self.assertIsNotNone(request.gr_features)
        # prompt_token_ids should be [0] (dummy) for empty history
        self.assertEqual(request.prompt_token_ids, [0])

    def test_extract_features_multi_key(self):
        """Test _extract_features with multiple keys."""
        keys, values, lengths = self.processor._extract_features(
            records=self.user_history,
            feature_names=["item_id", "timestamp", "rating"],
        )

        self.assertEqual(len(keys), 3)
        self.assertIn("item_id", keys)
        self.assertIn("timestamp", keys)
        self.assertIn("rating", keys)

        # Each key should have 3 values (one per record)
        for vals in values:
            self.assertEqual(len(vals), 3)

    def test_extract_features_missing_key(self):
        """Test that missing keys are excluded."""
        keys, values, lengths = self.processor._extract_features(
            records=[{"item_id": 1}, {"item_id": 2}],
            feature_names=["item_id", "nonexistent_key"],
        )

        # "nonexistent_key" should not appear since no record has it
        self.assertIn("item_id", keys)
        self.assertNotIn("nonexistent_key", keys)


class TestGROutputProcessor(unittest.TestCase):
    """Tests for GROutputProcessor output processor."""

    def setUp(self):
        """Set up test fixtures."""
        from vllm_gr.engine.gr_output_processor import GROutputProcessor

        self.processor = GROutputProcessor(
            default_top_k=3,
            log_stats=False,
        )

        self.candidates = [
            {"item_id": 10},
            {"item_id": 20},
            {"item_id": 30},
            {"item_id": 40},
            {"item_id": 50},
        ]

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        self.assertEqual(self.processor.default_top_k, 3)
        self.assertEqual(self.processor.log_stats, False)
        self.assertEqual(len(self.processor._active_requests), 0)

    def test_add_request(self):
        """Test adding a request to tracking."""
        self.processor.add_request(
            request_id="test-001",
            user_id="user_123",
            num_recommendations=3,
            original_candidates=self.candidates,
        )

        self.assertTrue(self.processor.has_unfinished_requests())
        self.assertEqual(self.processor.get_num_unfinished_requests(), 1)

    def test_remove_request(self):
        """Test removing a request."""
        self.processor.add_request(
            request_id="test-001",
            user_id="user_123",
            num_recommendations=3,
            original_candidates=self.candidates,
        )

        self.assertTrue(self.processor.has_unfinished_requests())

        self.processor.remove_request("test-001")

        self.assertFalse(self.processor.has_unfinished_requests())

    def test_create_gr_output_from_list_scores(self):
        """Test creating GR output from list scores."""
        self.processor.add_request(
            request_id="test-001",
            user_id="user_123",
            num_recommendations=3,
            original_candidates=self.candidates,
        )

        # Create mock output with scores
        scores = [0.1, 0.5, 0.3, 0.8, 0.2]  # item 40 has highest score
        output = self.processor._create_gr_output(
            request_id="test-001",
            user_id="user_123",
            scores=scores,
            candidates=self.candidates,
            num_recommendations=3,
        )

        self.assertIsNotNone(output)
        self.assertEqual(output.request_id, "test-001")
        self.assertEqual(output.user_id, "user_123")
        self.assertTrue(output.finished)

        # Top 3 should be: item 40 (0.8), item 20 (0.5), item 30 (0.3)
        self.assertEqual(len(output.recommendations), 3)
        self.assertEqual(output.recommendations[0].item_id, 40)
        self.assertEqual(output.recommendations[0].rank, 1)
        self.assertAlmostEqual(output.recommendations[0].score, 0.8)

    def test_top_k_limit(self):
        """Test that recommendations are limited to num_recommendations."""
        candidates = [{"item_id": i} for i in range(10)]
        scores = [0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15]

        output = self.processor._create_gr_output(
            request_id="test-002",
            user_id="user_456",
            scores=scores,
            candidates=candidates,
            num_recommendations=2,
        )

        # Should only have 2 recommendations
        self.assertEqual(len(output.recommendations), 2)
        # Top 2: item 1 (0.9), item 3 (0.8)
        self.assertEqual(output.recommendations[0].item_id, 1)
        self.assertEqual(output.recommendations[1].item_id, 3)


class TestGRProcessedOutputs(unittest.TestCase):
    """Tests for GRProcessedOutputs dataclass."""

    def test_creation(self):
        """Test creating GRProcessedOutputs."""
        from vllm_gr.engine.gr_output_processor import GRProcessedOutputs
        from vllm_gr.outputs import GRRequestOutput

        outputs = GRProcessedOutputs(
            request_outputs=[
                GRRequestOutput(
                    request_id="test-001",
                    user_id="user_1",
                    recommendations=[],
                ),
            ],
            iteration_stats={"total_time": 0.1},
        )

        self.assertEqual(len(outputs.request_outputs), 1)
        self.assertIsNotNone(outputs.iteration_stats)


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
            original_candidates=[{"item_id": 10}, {"item_id": 20}, {"item_id": 30}],
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

        # Serialize
        data = msgspec.msgpack.encode(payload)
        self.assertIsInstance(data, bytes)

        # Deserialize
        decoded = msgspec.msgpack.decode(data, type=GRFeaturesPayload)
        self.assertEqual(decoded.uih_keys, ["item_id"])
        self.assertEqual(decoded.uih_values_per_key[0], [1, 2, 3])
        self.assertEqual(decoded.user_id, "u1")


if __name__ == "__main__":
    unittest.main()
