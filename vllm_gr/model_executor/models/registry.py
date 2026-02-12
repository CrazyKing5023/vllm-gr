# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GR Model Registry.

This is the single source of truth for model registration in vllm-GR,
following the vllm-omni pattern.

Key components:
1. _GR_MODELS: Dictionary of GR-specific model architectures
2. GRModelRegistry: Merged registry containing vLLM + GR models
3. register_gr_models_to_vllm(): Registers GR models into vLLM's global registry

Usage:
    import vllm_gr  # Auto-registers on import
    
    # Or manually:
    from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
    register_gr_models_to_vllm()
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# vLLM Registry Imports
# ============================================================================

try:
    from vllm.model_executor.models.registry import (
        _VLLM_MODELS,
        _LazyRegisteredModel,
        _ModelRegistry,
        ModelRegistry as VLLMModelRegistry,
    )
    _HAS_VLLM_REGISTRY = True
except ImportError:
    _HAS_VLLM_REGISTRY = False
    _VLLM_MODELS = {}
    _LazyRegisteredModel = None
    _ModelRegistry = None
    VLLMModelRegistry = None

# ============================================================================
# GR Model Definitions
# ============================================================================

# GR-specific models: architecture -> (module_name, class_name)
# Module path is relative to vllm_gr.model_executor.models
# Note: We register multiple architecture names for flexibility:
# - HSTUEmbeddingModel: Follows vLLM's naming convention for embedding models
# - HSTUForRecommendation: Original name for backwards compatibility
_GR_MODELS: Dict[str, Tuple[str, str]] = {
    "HSTUEmbeddingModel": ("hstu_vllm", "HSTUForRecommendation"),
    "HSTUForRecommendation": ("hstu_vllm", "HSTUForRecommendation"),
}

# ============================================================================
# Registration State
# ============================================================================

_GR_MODELS_REGISTERED = False


def _create_gr_model_registry():
    """Create GR ModelRegistry that merges vLLM models with GR models."""
    if not _HAS_VLLM_REGISTRY:
        logger.warning("vLLM registry not available")
        return None
    
    registry_entries = {}
    
    # Add all vLLM built-in models
    for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items():
        registry_entries[model_arch] = _LazyRegisteredModel(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
    
    # Add GR-specific models
    for model_arch, (mod_name, cls_name) in _GR_MODELS.items():
        registry_entries[model_arch] = _LazyRegisteredModel(
            module_name=f"vllm_gr.model_executor.models.{mod_name}",
            class_name=cls_name,
        )
    
    return _ModelRegistry(registry_entries)


# Create the GR ModelRegistry (singleton)
GRModelRegistry = _create_gr_model_registry()


# ============================================================================
# Public API
# ============================================================================

def register_gr_models_to_vllm() -> bool:
    """
    Register GR models into vLLM's global ModelRegistry.
    
    This is the main registration function. Call this before model loading
    to ensure vLLM recognizes GR architectures.
    
    Returns:
        True if registration succeeded or already registered
    """
    global _GR_MODELS_REGISTERED
    
    if _GR_MODELS_REGISTERED:
        return True
    
    if not _HAS_VLLM_REGISTRY:
        logger.warning("vLLM registry not available")
        return False
    
    try:
        supported_archs = VLLMModelRegistry.get_supported_archs()
        
        for arch, (mod_name, cls_name) in _GR_MODELS.items():
            if arch not in supported_archs:
                # Use "module:class" format for lazy loading (avoids CUDA init)
                VLLMModelRegistry.register_model(
                    arch,
                    f"vllm_gr.model_executor.models.{mod_name}:{cls_name}"
                )
                logger.debug(f"Registered {arch} with vLLM ModelRegistry")
        
        _GR_MODELS_REGISTERED = True
        logger.info("GR models registered with vLLM ModelRegistry")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register GR models: {e}")
        return False


def is_registered() -> bool:
    """Check if GR models are registered with vLLM."""
    return _GR_MODELS_REGISTERED


def get_gr_architectures() -> list:
    """Get list of supported GR architectures."""
    return list(_GR_MODELS.keys())


def is_gr_model(arch: str) -> bool:
    """Check if an architecture is a GR model."""
    return arch in _GR_MODELS
