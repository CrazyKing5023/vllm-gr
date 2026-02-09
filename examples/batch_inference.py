#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Batch inference example for vllm-GR

This example demonstrates how to use GRLLM for batch recommendation.
It runs the full HSTU model forward pass. When no pretrained weights are provided,
the model uses randomly initialized weights for demonstration purposes.
"""

import sys
import os
import time
from typing import List

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from vllm_gr import GRLLM, GRPrompt, GRRequestOutput


def generate_sample_prompts(num_users: int, seq_length: int, num_candidates: int) -> List[GRPrompt]:
    """
    Generate sample GRPrompt requests for testing.
    
    Args:
        num_users: Number of users/requests to generate
        seq_length: Length of user history per user
        num_candidates: Number of candidates per user
    
    Returns:
        List of GRPrompt dictionaries
    """
    prompts: List[GRPrompt] = []
    
    for i in range(num_users):
        prompt: GRPrompt = {
            "user_id": f"user_{i:05d}",
            "user_history": [
                {"item_id": i * 1000 + j, "timestamp": 1609459200 + j * 3600}
                for j in range(seq_length)
            ],
            "candidates": [
                {"item_id": i * 10000 + k}
                for k in range(num_candidates)
            ],
        }
        prompts.append(prompt)
    
    return prompts


def main():
    """Run batch recommendation inference."""
    
    print("=" * 60)
    print(" vllm-GR Batch Inference Example")
    print("=" * 60)
    
    # 1. Initialize GRLLM
    # - model=None: No pretrained weights, creates model with random initialization
    # - Runs actual DlrmHSTU forward pass
    llm = GRLLM(
        model=None,  # No pretrained model path
        max_seq_len=64,
        max_candidates=50,
        default_top_k=5,
        # Model architecture parameters
        embedding_dim=64,
        hash_size=10000,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        attention_dim=32,
    )
    
    print(f"\n[GRLLM Initialized]")
    print(f"   Engine: {llm.engine_class.__name__}")
    print(f"   Model: Randomly initialized HSTU")
    
    # 2. Test parameters
    num_users = 10
    seq_length = 20
    num_candidates = 30
    top_k = 5
    
    print(f"\n[Batch Test Parameters]")
    print(f"   Number of users: {num_users}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Candidates per user: {num_candidates}")
    print(f"   Top-K: {top_k}")
    
    # 3. Generate GRPrompt requests
    prompts = generate_sample_prompts(num_users, seq_length, num_candidates)
    print(f"\n[Generated {len(prompts)} GRPrompt requests]")
    
    # 4. Method 1: Use recommend() with all prompts at once
    print(f"\n[Method 1: recommend() with all prompts]")
    print(f"   Running HSTU model inference...")
    start_time = time.time()
    outputs: List[GRRequestOutput] = llm.recommend(
        prompts=prompts,
        use_tqdm=False,
        top_k=top_k,
    )
    method1_time = (time.time() - start_time) * 1000
    print(f"   Time: {method1_time:.2f}ms")
    print(f"   Outputs: {len(outputs)}")
    
    # 5. Method 2: Use recommend_batch() for large batches
    print(f"\n[Method 2: recommend_batch() with batch_size=4]")
    start_time = time.time()
    outputs_batch: List[GRRequestOutput] = llm.recommend_batch(
        prompts=prompts,
        batch_size=4,
        top_k=top_k,
    )
    method2_time = (time.time() - start_time) * 1000
    print(f"   Time: {method2_time:.2f}ms")
    print(f"   Outputs: {len(outputs_batch)}")
    
    # 6. Display sample results
    print(f"\n[Sample Results (first 3 users)]")
    print(f"   (Note: Scores are from randomly initialized model)")
    for i, output in enumerate(outputs[:3]):
        top_items = [item.item_id for item in output.recommendations[:3]]
        top_scores = [f"{item.score:.3f}" for item in output.recommendations[:3]]
        print(f"\n   User {i}: {output.user_id}")
        print(f"     Request ID: {output.request_id}")
        print(f"     Top items: {top_items}")
        print(f"     Top scores: {top_scores}")
    
    # 7. Summary statistics
    print(f"\n[Performance Summary]")
    print(f"   Method 1 (recommend):       {method1_time:.2f}ms total, {method1_time/num_users:.2f}ms/request")
    print(f"   Method 2 (recommend_batch): {method2_time:.2f}ms total, {method2_time/num_users:.2f}ms/request")
    print(f"   Throughput: {num_users / (method1_time / 1000):.2f} requests/sec")
    
    # 8. Verify output structure
    print(f"\n[Output Structure Verification]")
    sample = outputs[0]
    print(f"   Type: {type(sample).__name__}")
    print(f"   Attributes: request_id, user_id, recommendations, finished, metrics")
    print(f"   Recommendation type: {type(sample.recommendations[0]).__name__}")
    print(f"   Item attributes: item_id, score, rank, features")
    
    # 9. Serialization example
    print(f"\n[Serialization Example]")
    output_dict = outputs[0].to_dict()
    print(f"   Keys: {list(output_dict.keys())}")
    print(f"   First recommendation: {output_dict['recommendations'][0]}")
    
    # 10. Cleanup
    llm.shutdown()
    
    print("\n" + "=" * 60)
    print(" Batch example completed successfully!")
    print(" Full HSTU model forward pass was executed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
