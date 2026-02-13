#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Simple inference example for vllm-GR

This example demonstrates how to use the GRLLM class for offline recommendation.
It runs the full HSTU model forward pass. When no pretrained weights are provided,
the model uses randomly initialized weights for demonstration purposes.

Demo Mode:
- Uses 'demo-model' directory which contains config.json with model architecture
- No actual weights file exists, so the model uses randomly initialized weights
- This allows testing the full inference pipeline without real model weights
"""

import sys
import os

# Enable CUDA debugging - this makes CUDA errors synchronous and provides better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Enable verbose logging
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from vllm_gr import GRLLM, GRPrompt

# Demo model path - relative to examples directory
DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "demo-model")


def main():
    """Run a simple recommendation inference."""
    
    print("=" * 60)
    print(" vllm-GR Simple Inference Example")
    print("=" * 60)
    
    # 1. Initialize GRLLM
    # 
    # Model Loading Modes:
    # - Production: GRLLM(model="path/to/pretrained_model")
    #   -> Loads config.json and weights from the directory
    # 
    # - Demo mode: GRLLM(model="path/to/demo-model")
    #   -> Loads config.json for architecture parameters
    #   -> Uses randomly initialized weights (no weights file present)
    #   -> Useful for validating the complete inference pipeline
    #
    # The demo-model directory contains:
    # - config.json: Model architecture configuration
    # - README.md: Documentation
    # (No weights file - model will be randomly initialized)
    
    print(f"\n[Model Path]")
    print(f"   Using demo model: {DEMO_MODEL_PATH}")
    
    llm = GRLLM(
        model=DEMO_MODEL_PATH,  # Demo model with config but no weights
        max_seq_len=64,         # Override config for demo (smaller for speed)
        max_candidates=50,
        default_top_k=10,
        # Override architecture for demo (smaller for speed)
        embedding_dim=64,
        hash_size=10000,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        attention_dim=32,
        # Scheduler configuration
        max_batch_size=16,
        scheduling_policy="fcfs",
    )
    
    print(f"\n[Configuration]")
    print(f"   Max sequence length: {llm.max_seq_len}")
    print(f"   Max candidates: {llm.max_candidates}")
    print(f"   Default top-k: {llm.default_top_k}")
    print(f"   Engine: {llm.engine_class.__name__}")
    print(f"   Model: HSTU with randomly initialized weights (demo mode)")
    
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
    # GRLLM.recommend() -> GRLLMEngine -> EngineCore -> HSTUForRecommendation -> DlrmHSTU
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
