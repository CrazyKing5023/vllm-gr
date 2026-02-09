# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GR Model Configuration.

Following vllm-omni pattern: Extend vLLM's ModelConfig and override
the registry property to use GRModelRegistry.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# Import vLLM's ModelConfig
try:
    from vllm.config import ModelConfig
    _HAS_VLLM_CONFIG = True
except ImportError:
    _HAS_VLLM_CONFIG = False
    ModelConfig = object  # type: ignore


@dataclass
class GRModelConfig(ModelConfig if _HAS_VLLM_CONFIG else object):
    """
    GR Model Configuration.
    
    Extends vLLM's ModelConfig with:
    1. Override registry to use GRModelRegistry
    2. GR-specific configuration fields
    
    This follows the vllm-omni pattern.
    """
    
    # GR-specific fields
    is_recommendation_model: bool = True
    max_seq_len: int = 512
    max_candidates: int = 1000
    num_tasks: int = 1
    task_names: List[str] = field(default_factory=lambda: ["is_click"])
    
    @property
    def registry(self):
        """Override registry to use GRModelRegistry."""
        from vllm_gr.model_executor.models.registry import GRModelRegistry
        return GRModelRegistry
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        if _HAS_VLLM_CONFIG and hasattr(super(), '__post_init__'):
            try:
                super().__post_init__()
            except Exception as e:
                # HSTU may not pass vLLM's LLM-specific validation
                logger.debug(f"Parent __post_init__ skipped: {e}")
    
    @classmethod
    def from_hstu_config(cls, hstu_config: Any, model_path: str = "", **kwargs):
        """Create GRModelConfig from HSTUConfig."""
        gr_fields = {
            "is_recommendation_model": True,
            "max_seq_len": getattr(hstu_config, 'max_seq_len', 512),
            "max_candidates": getattr(hstu_config, 'max_num_candidates', 1000),
            "task_names": getattr(hstu_config, 'task_names', ["is_click"]),
        }
        gr_fields["num_tasks"] = len(gr_fields["task_names"])
        gr_fields.update(kwargs)
        
        if _HAS_VLLM_CONFIG:
            return cls(model=model_path, **gr_fields)
        return cls(**gr_fields)
