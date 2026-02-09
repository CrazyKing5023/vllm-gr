#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Simple inference example for vllm-GR

This example demonstrates how to use the GRLLM class for offline recommendation.
It runs the full HSTU model forward pass. When no pretrained weights are provided,
the model uses randomly initialized weights for demonstration purposes.
"""

import sys
import os

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from vllm_gr import GRLLM, GRPrompt


def main():
    """Run a simple recommendation inference."""
    
    print("=" * 60)
    print(" vllm-GR Simple Inference Example")
    print("=" * 60)
    
    # 1. Initialize GRLLM
    # - For production: GRLLM(model="path/to/pretrained_hstu_model")
    # - For demo: model=None creates model with randomly initialized weights
    #
    # When model=None:
    # - Creates HSTUForRecommendation with randomly initialized weights
    # - Runs actual DlrmHSTU forward pass
    # - Useful for validating the complete inference pipeline
    llm = GRLLM(
        model=None,  # No pretrained model path
        max_seq_len=64,
        max_candidates=50,
        default_top_k=10,
        # Model architecture parameters
        embedding_dim=64,
        hash_size=10000,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        attention_dim=32,
    )
    
    print(f"\n[Configuration]")
    print(f"   Max sequence length: {llm.max_seq_len}")
    print(f"   Max candidates: {llm.max_candidates}")
    print(f"   Default top-k: {llm.default_top_k}")
    print(f"   Engine: {llm.engine_class.__name__}")
    print(f"   Model: Randomly initialized HSTU (no pretrained weights)")
    
    # 2. Prepare a recommendation request using GRPrompt format
    # GRPrompt is the standard input format for vllm-GR
    prompt: GRPrompt = {
        "user_id": "user_alice_12345",
        
        # User's interaction history
        "user_history": [
            {"item_id": 1001, "timestamp": 1609459200},  # The_Matrix
            {"item_id": 1002, "timestamp": 1609545600},  # Inception
            {"item_id": 1003, "timestamp": 1609632000},  # Interstellar
            {"item_id": 1004, "timestamp": 1609718400},  # Blade_Runner_2049
            {"item_id": 1005, "timestamp": 1609804800},  # Dune
            {"item_id": 1006, "timestamp": 1609891200},  # Arrival
            {"item_id": 1007, "timestamp": 1609977600},  # Ex_Machina
        ],
        
        # Candidate items to rank
        "candidates": [
            {"item_id": 2001},  # Tenet
            {"item_id": 2002},  # Avatar
            {"item_id": 2003},  # Edge_of_Tomorrow
            {"item_id": 2004},  # District_9
            {"item_id": 2005},  # Elysium
            {"item_id": 2006},  # Gravity
            {"item_id": 2007},  # The_Martian
            {"item_id": 2008},  # Annihilation
            {"item_id": 2009},  # Oblivion
            {"item_id": 2010},  # Looper
            {"item_id": 2011},  # Moon
            {"item_id": 2012},  # Sunshine
            {"item_id": 2013},  # Source_Code
            {"item_id": 2014},  # Prometheus
            {"item_id": 2015},  # Midnight_Sky
        ],
        
        # Optional: override default top_k
        "num_recommendations": 10,
    }
    
    print(f"\n[Request Details]")
    print(f"   User: {prompt['user_id']}")
    print(f"   History length: {len(prompt['user_history'])}")
    print(f"   Candidates: {len(prompt['candidates'])}")
    
    # 3. Get recommendations
    # This runs the full HSTU model forward pass:
    # GRLLM.recommend() -> GRLLMEngine -> GRModelRunner -> HSTUForRecommendation -> DlrmHSTU
    print(f"\n[Running HSTU model inference...]")
    outputs = llm.recommend(
        prompts=[prompt],
        use_tqdm=False,
        top_k=10,
    )
    
    # 4. Display results
    output = outputs[0]
    
    print(f"\n[Recommendation Results]")
    print(f"   Request ID: {output.request_id}")
    print(f"   User ID: {output.user_id}")
    print(f"   Finished: {output.finished}")
    print(f"\n   Top {len(output.recommendations)} Recommendations:")
    print(f"   (Note: Scores are from randomly initialized model)")
    
    for item in output.recommendations:
        print(f"   #{item.rank:2d}: item_{item.item_id:<10} (score: {item.score:.4f})")
    
    # 5. Access additional properties
    print(f"\n[Additional Output Properties]")
    print(f"   Top recommendation: {output.top_recommendation}")
    print(f"   All item IDs: {output.item_ids[:5]}...")
    print(f"   All scores: {[f'{s:.3f}' for s in output.scores[:5]]}...")
    
    # Convert to dict for JSON serialization
    output_dict = output.to_dict()
    print(f"   Serializable: {list(output_dict.keys())}")
    
    # 6. Cleanup
    llm.shutdown()
    
    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print(" Full HSTU model forward pass was executed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
