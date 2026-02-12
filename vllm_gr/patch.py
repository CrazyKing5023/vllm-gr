# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Monkey Patch mechanism for vllm-GR.

This module patches vLLM's Request class to support GRRequest.
The patch is applied automatically when vllm_gr is imported.
"""

import sys
import logging

logger = logging.getLogger("vllm_gr.patch")

# Save reference to original Request class
_OriginalRequest = None
_PATCHES_APPLIED = False
_PATCH_COUNT = 0

try:
    from vllm.v1.request import Request as _OriginalRequest
except ImportError:
    logger.debug("vllm_gr: vLLM v1 Request not found, trying alternative import")
    try:
        from vllm.request import Request as _OriginalRequest
    except ImportError:
        logger.debug("vllm_gr: vLLM Request not found, patches will be skipped")
        _OriginalRequest = None


def _apply_patches() -> int:
    """
    Apply Monkey Patches.
    
    Replace vLLM's Request class with GRRequest in vLLM modules.
    
    Returns:
        patched_count: Number of modules patched
    """
    global _PATCHES_APPLIED, _PATCH_COUNT
    
    if _PATCHES_APPLIED:
        return _PATCH_COUNT
    
    if _OriginalRequest is None:
        logger.warning("vllm_gr: vLLM not found, skipping patches")
        _PATCHES_APPLIED = True
        return 0
    
    # Import GR extension class
    from vllm_gr.request import GRRequest
    
    patched_count = 0
    patched_modules = []
    
    for module_name, module in list(sys.modules.items()):
        # Only process vllm modules
        if not module_name.startswith("vllm"):
            continue
        
        if module is None:
            continue
        
        # Replace Request class
        if hasattr(module, "Request"):
            if module.Request is _OriginalRequest:
                module.Request = GRRequest
                patched_count += 1
                patched_modules.append(module_name)
    
    _PATCHES_APPLIED = True
    _PATCH_COUNT = patched_count
    
    if patched_count > 0:
        logger.debug(f"vllm_gr: Applied {patched_count} monkey patches to: {patched_modules}")
    
    return patched_count


def get_original_request():
    """
    Get the original vLLM Request class.
    
    Returns:
        Original Request class, or None if vLLM is unavailable
    """
    return _OriginalRequest


def is_patched() -> bool:
    """
    Check if patches have been applied.
    
    Returns:
        bool: Whether patches have been applied
    """
    return _PATCHES_APPLIED


def get_patch_count() -> int:
    """
    Get the number of patched modules.
    
    Returns:
        int: Patch count
    """
    return _PATCH_COUNT


# Lazy patch execution, called explicitly in __init__.py
def apply_patches_if_needed() -> int:
    """
    Apply patches if needed.
    
    This function can be safely called multiple times.
    
    Returns:
        int: Patch count
    """
    return _apply_patches()
