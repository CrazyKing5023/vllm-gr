# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Monkey Patch mechanism for vllm-GR.

Following the vllm-omni pattern, this module patches vLLM's core types
to support GR-extended variants:
  - EngineCoreRequest → GREngineCoreRequest
  - Request → GRRequest
  - NewRequestData → GRNewRequestData

The patch is applied automatically when vllm_gr is imported, and can be
re-applied to cover late-loaded modules via re_apply_patches().
"""

import sys
import logging

logger = logging.getLogger("vllm_gr.patch")

# Save references to original classes
_OriginalEngineCoreRequest = None
_OriginalRequest = None
_OriginalNewRequestData = None

_PATCHES_APPLIED = False
_PATCH_COUNT = 0

try:
    from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
except ImportError:
    logger.debug("vllm_gr: vLLM EngineCoreRequest not found")

try:
    from vllm.v1.request import Request as _OriginalRequest
except ImportError:
    logger.debug("vllm_gr: vLLM Request not found")

try:
    from vllm.v1.core.sched.output import NewRequestData as _OriginalNewRequestData
except ImportError:
    logger.debug("vllm_gr: vLLM NewRequestData not found")


def _apply_patches(force: bool = False) -> int:
    """
    Apply Monkey Patches.

    Replace vLLM's core types with GR-extended versions in all loaded
    vLLM modules, following the vllm-omni pattern.

    Args:
        force: If True, re-scan sys.modules even if patches were already
               applied.  Use this to catch late-loaded modules (e.g.
               vllm.v1.engine.core loaded after initial import).

    Returns:
        patched_count: Number of module-level replacements performed
    """
    global _PATCHES_APPLIED, _PATCH_COUNT

    if _PATCHES_APPLIED and not force:
        return _PATCH_COUNT

    if _OriginalEngineCoreRequest is None and _OriginalRequest is None:
        logger.warning("vllm_gr: vLLM not found, skipping patches")
        _PATCHES_APPLIED = True
        return 0

    # Import GR extension classes
    from vllm_gr.engine import GREngineCoreRequest
    from vllm_gr.request import GRRequest
    from vllm_gr.core.sched.output import GRNewRequestData

    patched_count = 0
    patched_modules = []

    for module_name, module in list(sys.modules.items()):
        if "vllm" not in module_name:
            continue
        if module is None:
            continue

        # Patch EngineCoreRequest → GREngineCoreRequest
        if _OriginalEngineCoreRequest is not None:
            if hasattr(module, "EngineCoreRequest"):
                if module.EngineCoreRequest is _OriginalEngineCoreRequest:
                    module.EngineCoreRequest = GREngineCoreRequest
                    patched_count += 1
                    patched_modules.append(f"{module_name}.EngineCoreRequest")

        # Patch Request → GRRequest
        if _OriginalRequest is not None:
            if hasattr(module, "Request"):
                if module.Request is _OriginalRequest:
                    module.Request = GRRequest
                    patched_count += 1
                    patched_modules.append(f"{module_name}.Request")

        # Patch NewRequestData → GRNewRequestData
        if _OriginalNewRequestData is not None:
            if hasattr(module, "NewRequestData"):
                if module.NewRequestData is _OriginalNewRequestData:
                    module.NewRequestData = GRNewRequestData
                    patched_count += 1
                    patched_modules.append(f"{module_name}.NewRequestData")

    _PATCHES_APPLIED = True
    _PATCH_COUNT += patched_count

    if patched_count > 0:
        logger.info(
            f"vllm_gr: Applied {patched_count} monkey patches: "
            f"{patched_modules}"
        )

    return patched_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules.

    Call this at key entry points (e.g. GRLLMEngine.__init__) to ensure
    late-loaded modules like vllm.v1.engine.core are patched before the
    MsgpackDecoder is created.

    Returns:
        int: Number of NEW replacements performed in this call.
    """
    return _apply_patches(force=True)


def get_original_request():
    """Get the original vLLM Request class."""
    return _OriginalRequest


def get_original_engine_core_request():
    """Get the original vLLM EngineCoreRequest class."""
    return _OriginalEngineCoreRequest


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _PATCHES_APPLIED


def get_patch_count() -> int:
    """Get the number of patched module-level replacements."""
    return _PATCH_COUNT


def apply_patches_if_needed() -> int:
    """
    Apply patches if needed. Safe to call multiple times.

    Returns:
        int: Patch count
    """
    return _apply_patches()
