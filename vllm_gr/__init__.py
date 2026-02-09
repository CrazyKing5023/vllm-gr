# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vllm-GR: vLLM-based Generative Recommender Inference Engine

This package provides a vLLM-compatible interface for generative recommendation
models, specifically designed for DLRM_v3 (HSTU) architecture.

Main components:
- GRLLM: Main entry point for recommendation inference (inherits vLLM LLM)
- GRLLMEngine: Engine class (inherits vLLM LLMEngine)
- GRPrompt: User-friendly input format
- GRRequestOutput: Recommendation output format

vLLM Model Registration (following vllm-omni pattern):
- GRModelRegistry: Merges vLLM + GR models
- Models are automatically registered on import
- Use architectures=["HSTUForRecommendation"] in config.json
"""

__version__ = "0.1.0"

# Apply monkey patches first
from vllm_gr.patch import apply_patches_if_needed
_patch_count = apply_patches_if_needed()

# Register GR models with vLLM ModelRegistry
from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
_registered = register_gr_models_to_vllm()

# Core data types
from vllm_gr.inputs.data import (
    ItemInteraction,
    CandidateItem,
    GRPrompt,
    GRInputs,
    is_gr_prompt,
    gr_inputs,
)

from vllm_gr.outputs import (
    RecommendedItem,
    GRRequestOutput,
)

from vllm_gr.request import (
    GRRequest,
    is_gr_request,
)

# Engine components
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_processor import GRProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs
from vllm_gr.engine.gr_engine import GRLLMEngine
from vllm_gr.engine.model_runner import GRModelRunner

# Main entry point
from vllm_gr.entrypoint.gr import GRLLM

__all__ = [
    "__version__",
    # Core data types
    "ItemInteraction",
    "CandidateItem",
    "GRPrompt",
    "GRInputs",
    "is_gr_prompt",
    "gr_inputs",
    "RecommendedItem",
    "GRRequestOutput",
    "GRRequest",
    "is_gr_request",
    # Processors
    "GRProcessor",
    "GROutputProcessor",
    "GRProcessedOutputs",
    # Engine
    "GREngineArgs",
    "GRLLMEngine",
    "GRModelRunner",
    # Main entry point
    "GRLLM",
]
