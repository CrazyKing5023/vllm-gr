# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Model wrappers and registry for vllm-GR.

This module provides:
- HSTUForRecommendation: Unified HSTU model (vLLM compatible)
- HSTUModelOutput: Model output container
- register_gr_models_to_vllm(): Registers GR models with vLLM's ModelRegistry
"""

from vllm_gr.model_executor.models.hstu_vllm import (
    HSTUForRecommendation,
    HSTUModelOutput,
)
from vllm_gr.model_executor.models.registry import (
    register_gr_models_to_vllm,
    is_registered,
    get_gr_architectures,
    is_gr_model,
)

__all__ = [
    # Unified model
    "HSTUForRecommendation",
    "HSTUModelOutput",
    # Model registry
    "register_gr_models_to_vllm",
    "is_registered",
    "get_gr_architectures",
    "is_gr_model",
]
