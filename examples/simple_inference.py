#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Simple inference example for vllm-GR."""

import argparse
import os

# Enable verbose logging
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from runtime_utils import add_repo_root_to_path, add_runtime_args, describe_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)

# Demo model path - relative to examples directory
DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "demo-model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple vllm-GR recommendation example",
    )
    add_runtime_args(parser, default_model=DEMO_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum user history sequence length.")
    parser.add_argument("--max-candidates", type=int, default=50, help="Maximum number of candidate items.")
    return parser


def main():
    """Run a simple recommendation inference."""
    args = build_parser().parse_args()
    from vllm_gr import GRLLM, GRPrompt
    
    print("=" * 60)
    print(" vllm-GR Simple Inference Example")
    print("=" * 60)
    
    # 1. Initialize GRLLM
    #
    # Model architecture (layers, heads, embedding_dim, etc.) comes from
    # the model directory's config.json -- following the vLLM convention.
    # GRLLM only accepts runtime/serving parameters (max_seq_len, top_k, etc.).
    
    print(f"\n[Model Path]")
    print(f"   Using model: {args.model}")
    
    llm = GRLLM(
        model=args.model,
        max_seq_len=args.max_seq_len,
        max_candidates=args.max_candidates,
        default_top_k=args.top_k,
    )
    active_device = describe_runtime(llm, expected_device=args.expect_device)
    
    print(f"\n[Configuration]")
    print(f"   Max sequence length: {llm.max_seq_len}")
    print(f"   Max candidates: {llm.max_candidates}")
    print(f"   Default top-k: {llm.default_top_k}")
    print(f"   Engine: {llm.engine_class.__name__}")
    print(f"   Device path: {active_device}")
    print("   Model: HSTU with randomly initialized weights (demo mode)")
    
    # 2. Prepare a recommendation request using GRPrompt format
    # GRPrompt is the standard input format for vllm-GR.
    # Keep this aligned with examples/demo-model/config.json:
    #   UIH keys: item_id, timestamp
    #   candidate keys: item_candidate_id, query_time
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
        
        # Candidate items to rank (keys use model-internal candidate names)
        "candidates": [
            {"item_candidate_id": 2001, "query_time": 1610000000},
            {"item_candidate_id": 2002, "query_time": 1610000000},
            {"item_candidate_id": 2003, "query_time": 1610000000},
            {"item_candidate_id": 2004, "query_time": 1610000000},
            {"item_candidate_id": 2005, "query_time": 1610000000},
            {"item_candidate_id": 2006, "query_time": 1610000000},
            {"item_candidate_id": 2007, "query_time": 1610000000},
            {"item_candidate_id": 2008, "query_time": 1610000000},
            {"item_candidate_id": 2009, "query_time": 1610000000},
            {"item_candidate_id": 2010, "query_time": 1610000000},
            {"item_candidate_id": 2011, "query_time": 1610000000},
            {"item_candidate_id": 2012, "query_time": 1610000000},
            {"item_candidate_id": 2013, "query_time": 1610000000},
            {"item_candidate_id": 2014, "query_time": 1610000000},
            {"item_candidate_id": 2015, "query_time": 1610000000},
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
        top_k=args.top_k,
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
    print(f" Full HSTU model forward pass was executed on {active_device}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
