# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Configuration module for vllm-GR.

Provides:
- HSTUModelConfig: Internal dataclass configuration (detailed fields)
- HSTUConfig: HuggingFace-compatible PretrainedConfig subclass
- GRModelConfig: vLLM ModelConfig extension with GRModelRegistry
"""

from vllm_gr.config.hstu_config import (
    EmbeddingTableConfig,
    HSTUModelConfig,
    TaskConfig,
    MultitaskTaskType,
)
from vllm_gr.config.hstu_hf_config import HSTUConfig
from vllm_gr.config.model_config import GRModelConfig

__all__ = [
    # Internal configuration (detailed dataclass)
    "EmbeddingTableConfig",
    "HSTUModelConfig",
    "TaskConfig",
    "MultitaskTaskType",
    # HuggingFace-compatible configuration (for config.json parsing)
    "HSTUConfig",
    # vLLM ModelConfig extension
    "GRModelConfig",
]
