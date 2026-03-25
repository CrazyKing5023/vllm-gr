# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Monkey-patch mechanism for vllm-GR.

Replaces vLLM's core types with GR-extended variants in all loaded modules:
  - EngineCoreRequest → GREngineCoreRequest
  - Request → GRRequest
  - NewRequestData → GRNewRequestData

Activation:
  Primary — vLLM plugin system.  ``vllm_gr.plugin:register`` is registered
  under ``[project.entry-points."vllm.general_plugins"]`` in pyproject.toml.
  ``load_general_plugins()`` calls it in every process (main, EngineCore
  subprocess, Worker subprocess) *before* any MsgpackDecoder is created.

  Explicit main-process init — ``vllm_gr.plugin.initialize_runtime()`` is
  called by ``GREngineArgs.create_engine_config()`` before engine creation.

  Defense-in-depth — ``GRWorker.init_device()`` calls ``re_apply_patches()``
  to cover modules loaded after the initial plugin execution.

Safety features:
  - Source modules (vllm.v1.request, etc.) are patched first so that any
    future ``from X import Y`` in late-loaded modules picks up the GR variant.
  - verify_patches() checks the critical call-sites and logs actionable
    warnings when a patch is missing.
"""

import sys
import logging

logger = logging.getLogger("vllm_gr.patch")

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

# Source modules whose exported names are re-imported by consumer modules.
# Patching these first ensures late-loaded consumers get the GR variants.
_SOURCE_MODULES = {
    "EngineCoreRequest": "vllm.v1.engine",
    "Request": "vllm.v1.request",
    "NewRequestData": "vllm.v1.core.sched.output",
}

# Consumer modules that call from_engine_core_request / from_request.
_CRITICAL_CONSUMERS = {
    "EngineCoreRequest": ["vllm.v1.engine", "vllm.v1.engine.core"],
    "Request": ["vllm.v1.request", "vllm.v1.engine.core"],
    "NewRequestData": [
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.scheduler",
    ],
}


def _apply_patches(force: bool = False) -> int:
    """Replace vLLM types with GR-extended versions in all loaded modules.

    Patches source modules first (so future imports inherit the replacement),
    then patches all already-loaded consumer modules.

    Args:
        force: Re-scan sys.modules even if patches were already applied.

    Returns:
        Number of module-level replacements performed.
    """
    global _PATCHES_APPLIED, _PATCH_COUNT

    if _PATCHES_APPLIED and not force:
        return _PATCH_COUNT

    if _OriginalEngineCoreRequest is None and _OriginalRequest is None:
        logger.warning("vllm_gr: vLLM not found, skipping patches")
        _PATCHES_APPLIED = True
        return 0

    from vllm_gr.engine import GREngineCoreRequest
    from vllm_gr.request import GRRequest
    from vllm_gr.core.sched.output import GRNewRequestData

    replacements = [
        (_OriginalEngineCoreRequest, GREngineCoreRequest, "EngineCoreRequest"),
        (_OriginalRequest, GRRequest, "Request"),
        (_OriginalNewRequestData, GRNewRequestData, "NewRequestData"),
    ]

    patched_count = 0
    patched_modules = []

    # Phase 1: patch source modules so future `from X import Y` picks up GR.
    for original, replacement, attr_name in replacements:
        if original is None:
            continue
        src_mod_name = _SOURCE_MODULES.get(attr_name)
        src_mod = sys.modules.get(src_mod_name) if src_mod_name else None
        if src_mod is not None and getattr(src_mod, attr_name, None) is original:
            setattr(src_mod, attr_name, replacement)
            patched_count += 1
            patched_modules.append(f"{src_mod_name}.{attr_name}")

    # Phase 2: patch all loaded consumer modules.
    for module_name, module in list(sys.modules.items()):
        if module is None or "vllm" not in module_name:
            continue
        for original, replacement, attr_name in replacements:
            if original is None:
                continue
            if getattr(module, attr_name, None) is original:
                setattr(module, attr_name, replacement)
                patched_count += 1
                patched_modules.append(f"{module_name}.{attr_name}")

    _PATCHES_APPLIED = True
    _PATCH_COUNT += patched_count

    if patched_count > 0:
        logger.info(
            "vllm_gr: Applied %d monkey patches: %s",
            patched_count, patched_modules,
        )

    return patched_count


def verify_patches() -> list[str]:
    """Check that critical call-sites have been patched.

    Returns a list of warning messages for any unpatched sites.
    Safe to call at any time; does not modify anything.
    """
    from vllm_gr.engine import GREngineCoreRequest
    from vllm_gr.request import GRRequest
    from vllm_gr.core.sched.output import GRNewRequestData

    expected = {
        "EngineCoreRequest": GREngineCoreRequest,
        "Request": GRRequest,
        "NewRequestData": GRNewRequestData,
    }

    warnings = []
    for attr_name, expected_cls in expected.items():
        for mod_name in _CRITICAL_CONSUMERS.get(attr_name, []):
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            actual = getattr(mod, attr_name, None)
            if actual is not None and actual is not expected_cls:
                msg = (
                    f"vllm_gr: {mod_name}.{attr_name} is {actual.__name__}, "
                    f"expected {expected_cls.__name__}. "
                    f"gr_features may be lost for requests going through this module."
                )
                warnings.append(msg)
                logger.warning(msg)

    if not warnings:
        logger.debug("vllm_gr: All critical patches verified OK")

    return warnings


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules.

    Also runs verify_patches() and logs warnings for any gaps.
    """
    count = _apply_patches(force=True)
    verify_patches()
    return count


def get_original_request():
    return _OriginalRequest


def get_original_engine_core_request():
    return _OriginalEngineCoreRequest


def is_patched() -> bool:
    return _PATCHES_APPLIED


def get_patch_count() -> int:
    return _PATCH_COUNT


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
