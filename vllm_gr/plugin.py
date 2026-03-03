# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
vLLM general plugin entry point for vllm-GR.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This guarantees that the monkey-patches are applied **before** any
``MsgpackDecoder(EngineCoreRequest)`` is constructed, regardless of the
multiprocessing start method (spawn / fork / forkserver).
"""

import logging

logger = logging.getLogger("vllm_gr.plugin")


def register() -> None:
    """Apply vllm-GR monkey-patches and register models/configs.

    Designed to be called multiple times safely (idempotent).
    """
    from vllm_gr.patch import re_apply_patches
    from vllm_gr.config.hstu_hf_config import ensure_hstu_config_registered
    from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm

    count = re_apply_patches()
    ensure_hstu_config_registered()
    register_gr_models_to_vllm()

    logger.debug("vllm-GR plugin loaded (%d patches applied)", count)
