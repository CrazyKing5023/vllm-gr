# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GR Model Registry.

Registers GR-specific model architectures with vLLM's global ModelRegistry
using the stable public API (ModelRegistry.register_model).

Usage:
    from vllm_gr.plugin import initialize_runtime
    initialize_runtime()

    # Or manually:
    from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
    register_gr_models_to_vllm()
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# vLLM Registry Import (public API only)
# ============================================================================

try:
    from vllm.model_executor.models.registry import (
        ModelRegistry as VLLMModelRegistry,
    )
    _HAS_VLLM_REGISTRY = True
except ImportError:
    _HAS_VLLM_REGISTRY = False
    VLLMModelRegistry = None

# ============================================================================
# GR Model Definitions
# ============================================================================

# architecture -> (module_relative_name, class_name)
# Module path is relative to vllm_gr.model_executor.models
#
# "HSTUEmbeddingModel" is the canonical HuggingFace architecture name
# declared in HSTUConfig.architectures and every config.json.  vLLM's
# ModelRegistry resolves the model class by this name.  The "Embedding"
# suffix follows the vLLM convention for pooling-runner models.
_GR_MODELS: Dict[str, Tuple[str, str]] = {
    "HSTUEmbeddingModel": ("hstu_vllm", "HSTUForRecommendation"),
}

# ============================================================================
# Registration State
# ============================================================================

_GR_MODELS_REGISTERED = False


# ============================================================================
# Public API
# ============================================================================

def register_gr_models_to_vllm() -> bool:
    """
    Register GR models into vLLM's global ModelRegistry.

    Uses the stable ``ModelRegistry.register_model(arch, "module:class")``
    API which supports lazy loading and avoids premature CUDA init.

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
