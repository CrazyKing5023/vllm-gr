# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vLLM Model Registry for GR (re-exports from model_executor.models.registry).

This module provides a convenient top-level access to the GR model registry.
All actual implementation is in vllm_gr.model_executor.models.registry.

Usage:
    import vllm_gr  # Auto-registers on import
    
    # Or manually:
    from vllm_gr.registry import register_gr_models_to_vllm
    register_gr_models_to_vllm()
"""

# Re-export everything from the actual registry module
from vllm_gr.model_executor.models.registry import (
    GRModelRegistry,
    register_gr_models_to_vllm,
    is_registered,
    get_gr_architectures,
    is_gr_model,
)

# Backwards compatibility alias
register_hstu_model = register_gr_models_to_vllm
is_hstu_registered = is_registered
get_hstu_architectures = get_gr_architectures


def get_gr_model_registry():
    """Get the GR ModelRegistry instance."""
    return GRModelRegistry


__all__ = [
    # Primary API
    "GRModelRegistry",
    "register_gr_models_to_vllm",
    "is_registered",
    "get_gr_architectures",
    "is_gr_model",
    "get_gr_model_registry",
    # Backwards compatibility
    "register_hstu_model",
    "is_hstu_registered",
    "get_hstu_architectures",
]
