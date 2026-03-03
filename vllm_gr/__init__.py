# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""vllm-GR: vLLM-based Generative Recommender Inference Engine."""

__version__ = "0.3.0"

# Optional environment pre-check (opencv/numpy ABI conflicts).
# Runs before vllm is imported because vllm loads cv2 at module level.
from vllm_gr._env_check import check_opencv_env
check_opencv_env()

# Monkey-patch vLLM core types with GR-extended variants (must be first).
from vllm_gr.patch import apply_patches_if_needed
apply_patches_if_needed()

# Register HSTUConfig with AutoConfig & convertor registry.
from vllm_gr.config.hstu_hf_config import HSTUConfig, ensure_hstu_config_registered
ensure_hstu_config_registered()

# Register GR models with vLLM ModelRegistry.
from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
register_gr_models_to_vllm()

# Public API
from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.outputs import RecommendedItem, GRRequestOutput
from vllm_gr.request import GRRequest, is_gr_request
from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest
from vllm_gr.core.sched.output import GRNewRequestData
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRRequestState

# Engine
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_engine import GRLLMEngine
from vllm_gr.worker.gr_worker import GRWorker
from vllm_gr.worker.gr_model_runner import GRModelRunner
from vllm_gr.entrypoint.gr import GRLLM

__all__ = [
    "__version__",
    "HSTUConfig",
    "ensure_hstu_config_registered",
    "GRPrompt",
    "is_gr_prompt",
    "RecommendedItem",
    "GRRequestOutput",
    "GRRequest",
    "is_gr_request",
    "GRFeaturesPayload",
    "GREngineCoreRequest",
    "GRNewRequestData",
    "GRInputProcessor",
    "GROutputProcessor",
    "GRRequestState",
    # Engine
    "GREngineArgs",
    "GRLLMEngine",
    "GRWorker",
    "GRModelRunner",
    "GRLLM",
]
