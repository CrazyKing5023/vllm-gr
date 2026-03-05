#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Create a test model checkpoint for vllm-GR

This script creates a randomly initialized model checkpoint that can be used
for testing the inference pipeline without real pretrained weights.
"""

import os
import sys
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def create_test_model(output_dir: str, config_override: dict = None):
    """Create a test model checkpoint."""

    from vllm_gr.config.hstu_hf_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    print(f"Creating test model in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    config_dict = {
        "num_layers": 3,
        "num_heads": 4,
        "embedding_dim": 256,
        "attention_dim": 64,
        "hidden_dim": 128,
        "embedding_table_dim": 64,
        "preprocessor_hidden_dim": 128,
        "max_seq_len": 512,
        "max_num_candidates": 100,
        "hash_size": 100000,
        "feature_names": ["user_id", "item_id", "item_candidate_id"],
        "task_names": ["is_click", "is_like"],
        "is_inference": True,
    }

    if config_override:
        config_dict.update(config_override)

    config = HSTUConfig(**config_dict)

    config_path = os.path.join(output_dir, "config.json")
    config.save_pretrained(output_dir)
    print(f"  Config saved to: {config_path}")

    device = torch.device("cpu")
    model = HSTUForRecommendation.from_model_config(config, device)

    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.data.clone()

    weights_path = os.path.join(output_dir, "model.pt")
    torch.save(state_dict, weights_path)
    print(f"  Weights saved to: {weights_path}")

    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n  Model Summary:")
    print(f"     Total parameters: {total_params:,}")
    print(f"     Layers: {config.num_layers}")
    print(f"     Heads: {config.num_heads}")
    print(f"     Embedding dim: {config.embedding_dim}")
    print(f"     Tasks: {config.task_names}")

    print(f"\n  Test model created successfully!")
    print(f"   Use with: GRLLM(model='{output_dir}')")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create a test model checkpoint for vllm-GR")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_model",
        help="Output directory for the model checkpoint",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of HSTU layers",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=100000,
        help="Hash size for embedding tables",
    )

    args = parser.parse_args()

    config_override = {
        "num_layers": args.num_layers,
        "embedding_dim": args.embedding_dim,
        "hash_size": args.hash_size,
    }

    create_test_model(args.output_dir, config_override)


if __name__ == "__main__":
    main()
