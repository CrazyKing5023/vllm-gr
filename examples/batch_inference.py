#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Batch inference example for vllm-GR

This example demonstrates how to use GRLLM for batch recommendation.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_gr import GRLLM
from vllm_gr.config.hstu_config import HSTUModelConfig


def generate_sample_requests(num_users: int, seq_length: int, num_candidates: int):
    """Generate sample requests for testing."""
    requests = []
    
    for i in range(num_users):
        request = {
            "user_id": f"user_{i:05d}",
            "sequence": [
                {"item_id": f"item_{i}_{j}", "timestamp": 1609459200 + j * 3600}
                for j in range(seq_length)
            ],
            "candidates": [
                {"item_id": f"candidate_{i}_{k}"}
                for k in range(num_candidates)
            ],
        }
        requests.append(request)
    
    return requests


def main():
    """Run batch recommendation inference."""
    
    print("=" * 60)
    print(" vllm-GR Batch Inference Example")
    print("=" * 60)
    
    # Configuration
    config = HSTUModelConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=128,
        attention_dim=32,
        hidden_dim=64,
        embedding_table_dim=64,
        max_seq_len=128,
        max_num_candidates=50,
        hash_size=10000,
        task_names=["click", "purchase"],
    )
    
    # Initialize GRLLM
    llm = GRLLM(config=config, device="cuda")
    print(f"\n🚀 GRLLM initialized on {llm.device}")
    
    # Test parameters
    num_users = 10
    seq_length = 20
    num_candidates = 30
    top_k = 5
    
    print(f"\n📋 Batch Test Parameters:")
    print(f"   Number of users: {num_users}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Candidates per user: {num_candidates}")
    print(f"   Top-K: {top_k}")
    
    # Generate requests
    requests = generate_sample_requests(num_users, seq_length, num_candidates)
    print(f"\n📝 Generated {len(requests)} requests")
    
    # Run batch inference
    print(f"\n⏳ Running batch inference...")
    start_time = time.time()
    responses = llm.recommend_batch(requests, top_k=top_k)
    total_time = (time.time() - start_time) * 1000
    
    # Display results
    print(f"\n📊 Batch Results:")
    print(f"   Total time: {total_time:.2f}ms")
    print(f"   Avg time per request: {total_time / num_users:.2f}ms")
    print(f"   Throughput: {num_users / (total_time / 1000):.2f} requests/sec")
    
    print(f"\n   Sample results (first 3 users):")
    for i, response in enumerate(responses[:3]):
        print(f"\n   User {i}:")
        print(f"     Request ID: {response.request_id}")
        print(f"     Latency: {response.latency_ms:.2f}ms")
        print(f"     Top-{top_k}: {[r['item_id'] for r in response.recommendations[:3]]}...")
    
    # Summary statistics
    latencies = [r.latency_ms for r in responses]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print(f"\n📈 Latency Statistics:")
    print(f"   Average: {avg_latency:.2f}ms")
    print(f"   Min: {min_latency:.2f}ms")
    print(f"   Max: {max_latency:.2f}ms")
    
    print("\n" + "=" * 60)
    print(" Batch example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

