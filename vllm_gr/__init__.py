# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vllm-GR: vLLM-based Generative Recommender Inference Engine

This package provides a vLLM-compatible interface for generative recommendation
models, specifically designed for DLRM_v3 (HSTU) architecture.

Architecture:
- GRLLM: Main entry point (user interface)
- GRLLMEngine: Core engine orchestrating inference
- GRScheduler: Request scheduling and batching
- GRWorker/GRModelRunner: Model execution
- GRProcessor/GROutputProcessor: Input/output processing

Usage:
    >>> from vllm_gr import GRLLM
    >>> llm = GRLLM(model="path/to/hstu")
    >>> outputs = llm.recommend([{
    ...     "user_id": "user1",
    ...     "user_history": [{"item_id": 1}, {"item_id": 2}],
    ...     "candidates": [{"item_id": i} for i in range(100)],
    ... }])
"""

__version__ = "0.2.0"

# Apply monkey patches first
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

# Scheduler
from vllm_gr.core.gr_scheduler import GRScheduler, GRSchedulerConfig

# Worker components
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
    # Scheduler
    "GRScheduler",
    "GRSchedulerConfig",
    # Worker
    "GRWorker",
    "GRModelRunner",
    # Main entry point
    "GRLLM",
]
