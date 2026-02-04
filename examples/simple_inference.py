#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Simple inference example for vllm-GR

This example demonstrates how to use the GRLLM class for offline recommendation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_gr import GRLLM
from vllm_gr.config.hstu_config import HSTUModelConfig


def main():
    """Run a simple recommendation inference."""
    
    print("=" * 60)
    print(" vllm-GR Simple Inference Example")
    print("=" * 60)
    
    # 1. Create configuration
    # For production, you would load from a pretrained model path
    # Here we use a small config for demonstration
    config = HSTUModelConfig(
        num_layers=3,
        num_heads=4,
        embedding_dim=256,
        attention_dim=64,
        hidden_dim=128,
        embedding_table_dim=64,
        max_seq_len=256,
        max_num_candidates=50,
        hash_size=100000,
        task_names=["click", "like", "share"],
    )
    
    print(f"\n📋 Model Configuration:")
    print(f"   Layers: {config.num_layers}")
    print(f"   Heads: {config.num_heads}")
    print(f"   Embedding dim: {config.embedding_dim}")
    print(f"   Tasks: {config.task_names}")
    
    # 2. Initialize GRLLM
    # In production, use: GRLLM(model_path="path/to/pretrained/model")
    llm = GRLLM(config=config, device="cuda")
    print(f"\n🚀 GRLLM initialized on {llm.device}")
    
    # 3. Prepare a recommendation request
    user_id = "user_alice_12345"
    
    # User's interaction history
    sequence = [
        {"item_id": "The_Matrix_1999", "timestamp": 1609459200},
        {"item_id": "Inception_2010", "timestamp": 1609545600},
        {"item_id": "Interstellar_2014", "timestamp": 1609632000},
        {"item_id": "Blade_Runner_2049", "timestamp": 1609718400},
        {"item_id": "Dune_2021", "timestamp": 1609804800},
        {"item_id": "Arrival_2016", "timestamp": 1609891200},
        {"item_id": "Ex_Machina_2014", "timestamp": 1609977600},
    ]
    
    # Candidate items to rank
    candidates = [
        {"item_id": "Tenet_2020"},
        {"item_id": "Avatar_2009"},
        {"item_id": "Edge_of_Tomorrow_2014"},
        {"item_id": "District_9_2009"},
        {"item_id": "Elysium_2013"},
        {"item_id": "Gravity_2013"},
        {"item_id": "The_Martian_2015"},
        {"item_id": "Annihilation_2018"},
        {"item_id": "Oblivion_2013"},
        {"item_id": "Looper_2012"},
        {"item_id": "Moon_2009"},
        {"item_id": "Sunshine_2007"},
        {"item_id": "Source_Code_2011"},
        {"item_id": "Prometheus_2012"},
        {"item_id": "Midnight_Sky_2020"},
    ]
    
    print(f"\n📝 Request Details:")
    print(f"   User: {user_id}")
    print(f"   History length: {len(sequence)}")
    print(f"   Candidates: {len(candidates)}")
    
    # 4. Get recommendations
    print(f"\n⏳ Generating recommendations...")
    response = llm.recommend(
        user_id=user_id,
        sequence=sequence,
        candidates=candidates,
        top_k=10,
        request_id="example_request_001",
    )
    
    # 5. Display results
    print(f"\n📊 Recommendation Results:")
    print(f"   Request ID: {response.request_id}")
    print(f"   Latency: {response.latency_ms:.2f}ms")
    print(f"\n   Top {len(response.recommendations)} Recommendations:")
    
    for rec in response.recommendations:
        task_scores = rec.get("task_scores", {})
        task_str = ", ".join([f"{k}: {v:.3f}" for k, v in task_scores.items()])
        print(f"   #{rec['rank']:2d}: {rec['item_id']:<25} (score: {rec['score']:.4f}) [{task_str}]")
    
    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

