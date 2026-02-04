# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vllm-GR: vLLM-based Generative Recommender Inference Engine

Foundation version: Supports offline inference for DLRM_v3 (HSTU) model
"""

__version__ = "0.1.0"

from vllm_gr.entrypoint.gr import GRLLM, RecommendationResponse

__all__ = [
    "GRLLM",
    "RecommendationResponse",
]
