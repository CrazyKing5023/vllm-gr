# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Input and Output Processors

Tests for:
- GRProcessor: Input processor (GRPrompt -> GRRequest)
- GROutputProcessor: Output processor (scores -> GRRequestOutput)
- GRProcessedOutputs: Batch output container
"""

import os
import sys
import unittest
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGRProcessor(unittest.TestCase):
    """Tests for GRProcessor input processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        from vllm_gr.engine.gr_processor import GRProcessor
        
        self.processor = GRProcessor(
            max_seq_len=100,
            max_candidates=50,
        )
        
        # Sample user history
        self.user_history = [
            {"item_id": 1, "timestamp": 1000},
            {"item_id": 2, "timestamp": 2000},
            {"item_id": 3, "timestamp": 3000},
        ]
        
        # Sample candidates
        self.candidates = [
            {"item_id": 10},
            {"item_id": 20},
            {"item_id": 30},
            {"item_id": 40},
            {"item_id": 50},
        ]
    
    def test_processor_initialization(self):
        """Test processor initializes with correct parameters."""
        self.assertEqual(self.processor.max_seq_len, 100)
        self.assertEqual(self.processor.max_candidates, 50)
        self.assertEqual(self.processor.uih_feature_names, ["item_id"])
        self.assertEqual(self.processor.candidate_feature_names, ["candidate_item_id"])
    
    def test_process_gr_inputs_basic(self):
        """Test basic input processing."""
        from vllm_gr.inputs.data import GRPrompt
        
        prompt: GRPrompt = {
            "user_id": "user_123",
            "user_history": self.user_history,
            "candidates": self.candidates,
        }
        
        gr_request = self.processor.process_gr_inputs(
            request_id="test-001",
            prompt=prompt,
            num_recommendations=3,
        )
        
        # Verify request properties
        self.assertEqual(gr_request.request_id, "test-001")
        self.assertEqual(gr_request.user_id, "user_123")
        self.assertEqual(gr_request.num_recommendations, 3)
        self.assertEqual(len(gr_request.original_candidates), 5)
    
    def test_process_gr_inputs_with_context(self):
        """Test input processing with context."""
        from vllm_gr.inputs.data import GRPrompt
        
        prompt: GRPrompt = {
            "user_id": 456,
            "user_history": self.user_history,
            "candidates": self.candidates,
            "context": {"device": "mobile", "time_of_day": "morning"},
        }
        
        gr_request = self.processor.process_gr_inputs(
            request_id="test-002",
            prompt=prompt,
            num_recommendations=5,
        )
        
        # Verify context is passed through
        self.assertIsNotNone(gr_request.gr_inputs)
        self.assertIn("context", gr_request.gr_inputs)
        self.assertEqual(gr_request.gr_inputs["context"]["device"], "mobile")
    
    def test_encode_user_history(self):
        """Test user history encoding."""
        kjt = self.processor._encode_user_history(self.user_history)
        
        # Verify encoding result
        if isinstance(kjt, dict):
            self.assertIn("keys", kjt)
            self.assertIn("values", kjt)
            self.assertIn("lengths", kjt)
            self.assertEqual(len(kjt["values"]), 3)
            self.assertEqual(kjt["values"], [1, 2, 3])
    
    def test_encode_candidates(self):
        """Test candidate encoding."""
        kjt = self.processor._encode_candidates(self.candidates)
        
        # Verify encoding result
        if isinstance(kjt, dict):
            self.assertIn("keys", kjt)
            self.assertIn("values", kjt)
            self.assertIn("lengths", kjt)
            self.assertEqual(len(kjt["values"]), 5)
            self.assertEqual(kjt["values"], [10, 20, 30, 40, 50])
    
    def test_max_seq_len_truncation(self):
        """Test that long sequences are truncated."""
        from vllm_gr.engine.gr_processor import GRProcessor
        processor = GRProcessor(max_seq_len=2)
        
        # Only last 2 items should be kept
        long_history = [
            {"item_id": 1, "timestamp": 1000},
            {"item_id": 2, "timestamp": 2000},
            {"item_id": 3, "timestamp": 3000},
            {"item_id": 4, "timestamp": 4000},
        ]
        
        kjt = processor._encode_user_history(long_history)
        
        if isinstance(kjt, dict):
            # Should contain only last 2 items
            self.assertEqual(len(kjt["values"]), 2)
            self.assertEqual(kjt["values"], [3, 4])
    
    def test_max_candidates_truncation(self):
        """Test that candidates are truncated to max_candidates."""
        from vllm_gr.engine.gr_processor import GRProcessor
        processor = GRProcessor(max_candidates=2)
        
        kjt = processor._encode_candidates(self.candidates)
        
        if isinstance(kjt, dict):
            # Should contain only first 2 items
            self.assertEqual(len(kjt["values"]), 2)
            self.assertEqual(kjt["values"], [10, 20])
    
    def test_empty_history(self):
        """Test handling of empty user history."""
        kjt = self.processor._encode_user_history([])
        
        if isinstance(kjt, dict):
            self.assertEqual(len(kjt["values"]), 0)
    
    def test_empty_candidates(self):
        """Test handling of empty candidates."""
        kjt = self.processor._encode_candidates([])
        
        if isinstance(kjt, dict):
            self.assertEqual(len(kjt["values"]), 0)


class TestGROutputProcessor(unittest.TestCase):
    """Tests for GROutputProcessor output processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        from vllm_gr.engine.gr_output_processor import GROutputProcessor
        from vllm_gr.request import GRRequest
        
        self.processor = GROutputProcessor(
            default_top_k=3,
            log_stats=False,
        )
        
        # Create a sample request
        self.sample_request = GRRequest(
            request_id="test-001",
            user_id="user_123",
            num_recommendations=3,
            original_candidates=[
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
                {"item_id": 40},
                {"item_id": 50},
            ],
        )
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        self.assertEqual(self.processor.default_top_k, 3)
        self.assertEqual(self.processor.log_stats, False)
        self.assertEqual(len(self.processor._active_requests), 0)
    
    def test_add_gr_request(self):
        """Test adding a request to tracking."""
        self.processor.add_gr_request(
            self.sample_request,
            original_candidates=[{"item_id": 10}, {"item_id": 20}],
        )
        
        self.assertTrue(self.processor.has_unfinished_requests())
        self.assertEqual(self.processor.get_num_unfinished_requests(), 1)
    
    def test_remove_request(self):
        """Test removing a request."""
        self.processor.add_gr_request(
            self.sample_request,
            original_candidates=[{"item_id": 10}],
        )
        
        self.assertTrue(self.processor.has_unfinished_requests())
        
        self.processor.remove_request("test-001")
        
        self.assertFalse(self.processor.has_unfinished_requests())
    
    def test_create_output_from_scores_list(self):
        """Test creating output from score list."""
        self.processor.add_gr_request(
            self.sample_request,
            original_candidates=self.sample_request.original_candidates,
        )
        
        # Create mock scores (list format)
        scores = [0.1, 0.5, 0.3, 0.8, 0.2]  # item 40 has highest score
        
        output = self.processor.create_output_from_scores(
            request_id="test-001",
            scores=scores,
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
    
    def test_create_output_removes_request(self):
        """Test that creating output removes the request from tracking."""
        self.processor.add_gr_request(
            self.sample_request,
            original_candidates=self.sample_request.original_candidates,
        )
        
        self.assertTrue(self.processor.has_unfinished_requests())
        
        self.processor.create_output_from_scores(
            request_id="test-001",
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        
        self.assertFalse(self.processor.has_unfinished_requests())
    
    def test_top_k_limit(self):
        """Test that recommendations are limited to num_recommendations."""
        from vllm_gr.request import GRRequest
        
        request = GRRequest(
            request_id="test-002",
            user_id="user_456",
            num_recommendations=2,  # Only want top 2
            original_candidates=[{"item_id": i} for i in range(10)],
        )
        
        self.processor.add_gr_request(
            request,
            original_candidates=request.original_candidates,
        )
        
        scores = [0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15]
        
        output = self.processor.create_output_from_scores(
            request_id="test-002",
            scores=scores,
        )
        
        # Should only have 2 recommendations
        self.assertEqual(len(output.recommendations), 2)
        # Top 2: item 1 (0.9), item 3 (0.8)
        self.assertEqual(output.recommendations[0].item_id, 1)
        self.assertEqual(output.recommendations[1].item_id, 3)
    
    def test_unknown_request_returns_none(self):
        """Test that unknown request ID returns None."""
        output = self.processor.create_output_from_scores(
            request_id="unknown-id",
            scores=[0.1, 0.2, 0.3],
        )
        
        self.assertIsNone(output)


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


if __name__ == "__main__":
    unittest.main()
