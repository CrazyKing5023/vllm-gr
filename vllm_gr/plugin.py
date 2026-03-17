# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""vLLM general plugin entry point for vllm-GR.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This module is the formal runtime initialization boundary for vllm-GR.
It is responsible only for:

- applying GR monkey patches
- registering HSTU config hooks
- registering GR models with vLLM
"""

import logging

logger = logging.getLogger("vllm_gr.plugin")


def initialize_runtime() -> dict[str, int | bool]:
    """Initialize the current process for vllm-GR runtime usage.

    Safe to call repeatedly. The underlying patch and registration helpers are
    idempotent, so this can be used both by the vLLM plugin loader and by
    explicit engine startup paths in the main process.
    """
    from vllm_gr.patch import re_apply_patches
    from vllm_gr.config.hstu_hf_config import ensure_hstu_config_registered
    from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm

    patch_count = re_apply_patches()
    config_registered = ensure_hstu_config_registered()
    models_registered = register_gr_models_to_vllm()

    state = {
        "patch_count": patch_count,
        "config_registered": config_registered,
        "models_registered": models_registered,
    }
    logger.debug(
        "vllm-GR runtime initialized: patches=%d, config_registered=%s, "
        "models_registered=%s",
        patch_count,
        config_registered,
        models_registered,
    )
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    state = initialize_runtime()
    logger.debug("vllm-GR plugin loaded (%d patches applied)", state["patch_count"])
