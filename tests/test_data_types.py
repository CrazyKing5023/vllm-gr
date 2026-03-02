# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unit Tests: Core Data Types

Tests for:
- GRPrompt data types
- GRRequestOutput, RecommendedItem output types
- GRRequest extended request class
- Monkey Patch mechanism
- Module exports
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGRPrompt(unittest.TestCase):
    """Tests for GRPrompt input type"""
    
    def test_basic_gr_prompt(self):
        """Test basic GRPrompt creation"""
        from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
        
        prompt: GRPrompt = {
            "user_id": 123,
            "user_history": [
                {"item_id": 1},
                {"item_id": 2},
                {"item_id": 3},
            ],
            "candidates": [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
            ],
        }
        
        self.assertTrue(is_gr_prompt(prompt))
        self.assertEqual(prompt["user_id"], 123)
        self.assertEqual(len(prompt["user_history"]), 3)
        self.assertEqual(len(prompt["candidates"]), 3)
    
    def test_gr_prompt_with_optional_fields(self):
        """Test GRPrompt with optional fields"""
        from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
        
        prompt: GRPrompt = {
            "user_id": "user_abc",
            "user_history": [
                {"item_id": 1, "timestamp": 1000, "rating": 5, "action_type": 1},
                {"item_id": 2, "timestamp": 2000, "rating": 4, "action_type": 2},
            ],
            "candidates": [
                {"item_id": 10, "category_id": 1},
                {"item_id": 20, "category_id": 2},
            ],
            "num_recommendations": 5,
            "context": {"device_type": 0, "time_of_day": 1},
        }
        
        self.assertTrue(is_gr_prompt(prompt))
        self.assertEqual(prompt["num_recommendations"], 5)
        self.assertEqual(prompt["context"]["device_type"], 0)
    
    def test_is_gr_prompt_false_cases(self):
        """Test is_gr_prompt returns False for invalid inputs"""
        from vllm_gr.inputs.data import is_gr_prompt
        
        # Not a dict
        self.assertFalse(is_gr_prompt("not a dict"))
        self.assertFalse(is_gr_prompt(123))
        self.assertFalse(is_gr_prompt(None))
        
        # Missing required fields
        self.assertFalse(is_gr_prompt({"user_id": 1}))
        self.assertFalse(is_gr_prompt({"user_id": 1, "user_history": []}))
        self.assertFalse(is_gr_prompt({"user_id": 1, "candidates": []}))
        
        # Empty dict
        self.assertFalse(is_gr_prompt({}))


class TestItemInteraction(unittest.TestCase):
    """Tests for ItemInteraction type"""
    
    def test_minimal_interaction(self):
        """Test minimal ItemInteraction with only item_id"""
        from vllm_gr.inputs.data import ItemInteraction
        
        interaction: ItemInteraction = {"item_id": 42}
        self.assertEqual(interaction["item_id"], 42)
    
    def test_full_interaction(self):
        """Test ItemInteraction with all optional fields"""
        from vllm_gr.inputs.data import ItemInteraction
        
        interaction: ItemInteraction = {
            "item_id": 42,
            "timestamp": 1609459200,
            "rating": 5,
            "action_type": 2,
            "category_id": 10,
        }
        
        self.assertEqual(interaction["item_id"], 42)
        self.assertEqual(interaction["timestamp"], 1609459200)
        self.assertEqual(interaction["rating"], 5)


class TestRecommendedItem(unittest.TestCase):
    """Tests for RecommendedItem output type"""
    
    def test_recommended_item_creation(self):
        """Test RecommendedItem creation"""
        from vllm_gr.outputs import RecommendedItem
        
        item = RecommendedItem(
            item_id=42,
            score=0.95,
            rank=1,
        )
        
        self.assertEqual(item.item_id, 42)
        self.assertEqual(item.score, 0.95)
        self.assertEqual(item.rank, 1)
        self.assertIsNone(item.features)
    
    def test_recommended_item_with_features(self):
        """Test RecommendedItem with features"""
        from vllm_gr.outputs import RecommendedItem
        
        item = RecommendedItem(
            item_id="item_abc",
            score=0.85,
            rank=2,
            features={"category": "electronics", "price": 99.99},
        )
        
        self.assertEqual(item.item_id, "item_abc")
        self.assertEqual(item.features["category"], "electronics")
    
    def test_recommended_item_to_dict(self):
        """Test RecommendedItem to_dict method"""
        from vllm_gr.outputs import RecommendedItem
        
        item = RecommendedItem(
            item_id=42,
            score=0.95,
            rank=1,
            features={"key": "value"},
        )
        
        d = item.to_dict()
        self.assertEqual(d["item_id"], 42)
        self.assertEqual(d["score"], 0.95)
        self.assertEqual(d["rank"], 1)
        self.assertEqual(d["features"]["key"], "value")


class TestGRRequestOutput(unittest.TestCase):
    """Tests for GRRequestOutput"""
    
    def test_gr_request_output_creation(self):
        """Test GRRequestOutput creation"""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem
        
        recommendations = [
            RecommendedItem(item_id=1, score=0.9, rank=1),
            RecommendedItem(item_id=2, score=0.8, rank=2),
            RecommendedItem(item_id=3, score=0.7, rank=3),
        ]
        
        output = GRRequestOutput(
            request_id="req_001",
            user_id=123,
            recommendations=recommendations,
        )
        
        self.assertEqual(output.request_id, "req_001")
        self.assertEqual(output.user_id, 123)
        self.assertEqual(len(output.recommendations), 3)
        self.assertTrue(output.finished)
    
    def test_gr_request_output_properties(self):
        """Test GRRequestOutput properties"""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem
        
        recommendations = [
            RecommendedItem(item_id=10, score=0.95, rank=1),
            RecommendedItem(item_id=20, score=0.85, rank=2),
        ]
        
        output = GRRequestOutput(
            request_id="req_002",
            user_id="user_abc",
            recommendations=recommendations,
        )
        
        # Test properties
        self.assertEqual(output.top_recommendation.item_id, 10)
        self.assertEqual(output.item_ids, [10, 20])
        self.assertEqual(output.scores, [0.95, 0.85])
    
    def test_gr_request_output_to_dict(self):
        """Test GRRequestOutput to_dict method"""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem
        
        recommendations = [
            RecommendedItem(item_id=1, score=0.9, rank=1),
        ]
        
        output = GRRequestOutput(
            request_id="req_003",
            user_id=123,
            recommendations=recommendations,
            metrics={"latency_ms": 10.5},
        )
        
        d = output.to_dict()
        self.assertEqual(d["request_id"], "req_003")
        self.assertEqual(d["user_id"], 123)
        self.assertEqual(len(d["recommendations"]), 1)
        self.assertEqual(d["recommendations"][0]["item_id"], 1)
        self.assertEqual(d["metrics"]["latency_ms"], 10.5)
    
    def test_gr_request_output_repr(self):
        """Test GRRequestOutput __repr__"""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem
        
        output = GRRequestOutput(
            request_id="req_004",
            user_id=123,
            recommendations=[RecommendedItem(item_id=1, score=0.9, rank=1)],
        )
        
        repr_str = repr(output)
        self.assertIn("req_004", repr_str)
        self.assertIn("123", repr_str)
        self.assertIn("num_recommendations=1", repr_str)


class TestGRRequest(unittest.TestCase):
    """Tests for GRRequest (extended vLLM Request with gr_features)"""
    
    def test_gr_request_without_features(self):
        """Test GRRequest creation without GR features"""
        from vllm_gr.request import GRRequest, is_gr_request
        
        # GRRequest without gr_features behaves like a normal Request
        request = GRRequest(
            request_id="req1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=None,
            eos_token_id=None,
        )
        
        self.assertEqual(request.request_id, "req1")
        self.assertIsNone(request.gr_features)
        self.assertFalse(is_gr_request(request))
    
    def test_gr_request_with_features(self):
        """Test GRRequest creation with GR features"""
        from vllm_gr.request import GRRequest, is_gr_request
        from vllm_gr.engine import GRFeaturesPayload
        
        payload = GRFeaturesPayload(
            uih_keys=["item_id"],
            uih_values_per_key=[[1, 2, 3]],
            uih_lengths_per_key=[[1, 1, 1]],
            cand_keys=["item_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[[1, 1]],
            user_id="user_123",
            num_recommendations=5,
        )
        
        request = GRRequest(
            gr_features=payload,
            request_id="req2",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=None,
            eos_token_id=None,
        )
        
        self.assertEqual(request.request_id, "req2")
        self.assertIsNotNone(request.gr_features)
        self.assertEqual(request.gr_features.user_id, "user_123")
        self.assertTrue(is_gr_request(request))
    
    def test_is_gr_request_function(self):
        """Test is_gr_request standalone function"""
        from vllm_gr.request import is_gr_request
        
        # Non-GRRequest objects should return False
        self.assertFalse(is_gr_request(None))
        self.assertFalse(is_gr_request({}))
        self.assertFalse(is_gr_request("string"))


class TestMonkeyPatch(unittest.TestCase):
    """Tests for Monkey Patch mechanism"""
    
    def test_patch_module_imports(self):
        """Test that patch module can be imported"""
        from vllm_gr.patch import (
            apply_patches_if_needed,
            is_patched,
            get_patch_count,
            get_original_request,
        )
        
        # Functions should be callable
        self.assertTrue(callable(apply_patches_if_needed))
        self.assertTrue(callable(is_patched))
        self.assertTrue(callable(get_patch_count))
        self.assertTrue(callable(get_original_request))
    
    def test_patch_idempotent(self):
        """Test that patch can be applied multiple times safely"""
        from vllm_gr.patch import apply_patches_if_needed
        
        count1 = apply_patches_if_needed()
        count2 = apply_patches_if_needed()
        
        # Should return same count (idempotent)
        self.assertEqual(count1, count2)
    
    def test_patch_status(self):
        """Test patch status functions"""
        from vllm_gr.patch import is_patched, apply_patches_if_needed
        
        # Apply patches
        apply_patches_if_needed()
        
        # Should be patched (or attempted)
        self.assertTrue(is_patched())


class TestModuleExports(unittest.TestCase):
    """Tests for module exports from vllm_gr package"""
    
    def test_main_exports(self):
        """Test that main types are exported from vllm_gr"""
        import vllm_gr
        
        # Core types should be accessible
        self.assertTrue(hasattr(vllm_gr, "GRPrompt"))
        self.assertTrue(hasattr(vllm_gr, "GRRequestOutput"))
        self.assertTrue(hasattr(vllm_gr, "RecommendedItem"))
        self.assertTrue(hasattr(vllm_gr, "GRRequest"))
        
        # Helper functions should be accessible
        self.assertTrue(hasattr(vllm_gr, "is_gr_prompt"))
        self.assertTrue(hasattr(vllm_gr, "is_gr_request"))
        
        # Version should be accessible
        self.assertTrue(hasattr(vllm_gr, "__version__"))
    
    def test_inputs_submodule(self):
        """Test inputs submodule exports"""
        from vllm_gr.inputs import (
            ItemInteraction,
            CandidateItem,
            GRPrompt,
            is_gr_prompt,
        )
        
        # All should be importable
        self.assertIsNotNone(ItemInteraction)
        self.assertIsNotNone(CandidateItem)
        self.assertIsNotNone(GRPrompt)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" Unit Tests: Core Data Types")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGRPrompt))
    suite.addTests(loader.loadTestsFromTestCase(TestItemInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendedItem))
    suite.addTests(loader.loadTestsFromTestCase(TestGRRequestOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestGRRequest))
    suite.addTests(loader.loadTestsFromTestCase(TestMonkeyPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleExports))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
