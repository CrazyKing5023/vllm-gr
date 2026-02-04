# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vllm-GR: vLLM-based Generative Recommender Inference Engine

基础版本：支持 DLRM_v3 (HSTU) 模型的离线推理
"""

__version__ = "0.1.0"

from vllm_gr.entrypoint.gr import GRLLM, RecommendationResponse

__all__ = [
    "GRLLM",
    "RecommendationResponse",
]

