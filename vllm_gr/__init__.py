# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vllm-GR: vLLM-based Generative Recommender Inference Engine

This package provides a vLLM-compatible interface for generative recommendation
models, specifically designed for DLRM_v3 (HSTU) architecture.

Architecture (following vllm-omni pattern):
- GRLLM: Main entry point (user interface)
- GRLLMEngine: Core engine orchestrating inference via vLLM EngineCore
- GRInputProcessor: Converts GRPrompt → GREngineCoreRequest (with full features)
- GROutputProcessor: Converts engine outputs → GRRequestOutput (Top-K)
- GRWorker + GRModelRunner: Custom worker/runner for GR feature handling
- Monkey Patching: Extends EngineCoreRequest, Request, NewRequestData

Data Flow:
  GRPrompt → GRInputProcessor → GREngineCoreRequest(gr_features)
    → EngineCore → Scheduler → GRNewRequestData(gr_features)
    → GRModelRunner._update_states (decode features)
    → GRModelRunner._preprocess (embed + inject kwargs into model_kwargs)
    → HSTUForRecommendation.forward()
    → GROutputProcessor → GRRequestOutput

Usage:
    >>> from vllm_gr import GRLLM
    >>> llm = GRLLM(model="path/to/hstu")
    >>> outputs = llm.recommend([{
    ...     "user_id": "user1",
    ...     "user_history": [
    ...         {"item_id": 1, "timestamp": 1000, "rating": 5},
    ...         {"item_id": 2, "timestamp": 2000, "rating": 3},
    ...     ],
    ...     "candidates": [{"item_id": i} for i in range(100)],
    ... }])
"""

__version__ = "0.3.0"

# Apply monkey patches first (MUST happen before any vLLM usage)
# This replaces EngineCoreRequest, Request, NewRequestData with GR-extended versions
from vllm_gr.patch import apply_patches_if_needed
_patch_count = apply_patches_if_needed()

# Register HSTUConfig with HuggingFace AutoConfig
# This must happen BEFORE any vLLM engine config is created
from vllm_gr.config.hstu_hf_config import (
    HSTUConfig,
    ensure_hstu_config_registered,
)
_hstu_registered = ensure_hstu_config_registered()

# Register GR models with vLLM ModelRegistry
from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
_models_registered = register_gr_models_to_vllm()

# Core data types
from vllm_gr.inputs.data import (
    ItemInteraction,
    CandidateItem,
    GRPrompt,
    is_gr_prompt,
)

from vllm_gr.outputs import (
    RecommendedItem,
    GRRequestOutput,
)

from vllm_gr.request import (
    GRRequest,
    is_gr_request,
)

# Pipeline data types
from vllm_gr.engine import (
    GRFeaturesPayload,
    GREngineCoreRequest,
)
from vllm_gr.core.sched.output import GRNewRequestData

# Processors
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs

# Engine
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_engine import GRLLMEngine

# Worker
from vllm_gr.worker.gr_worker import GRWorker
from vllm_gr.worker.gr_model_runner import GRModelRunner

# Main entry point
from vllm_gr.entrypoint.gr import GRLLM

__all__ = [
    "__version__",
    # Config
    "HSTUConfig",
    "ensure_hstu_config_registered",
    # Core data types
    "ItemInteraction",
    "CandidateItem",
    "GRPrompt",
    "is_gr_prompt",
    "RecommendedItem",
    "GRRequestOutput",
    "GRRequest",
    "is_gr_request",
    # Pipeline data types
    "GRFeaturesPayload",
    "GREngineCoreRequest",
    "GRNewRequestData",
    # Processors
    "GRInputProcessor",
    "GROutputProcessor",
    "GRProcessedOutputs",
    # Engine
    "GREngineArgs",
    "GRLLMEngine",
    # Worker
    "GRWorker",
    "GRModelRunner",
    # Main entry point
    "GRLLM",
]
