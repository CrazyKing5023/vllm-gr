# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Static HSTU model family registration (layer-1).

Pure Python — no vLLM or heavy imports.  Registers the ``hstu`` model
family with the global ``_GR_MODEL_FAMILIES`` dict so that
``arg_utils.py`` can resolve worker classes before vLLM is fully loaded.
"""

from __future__ import annotations

from typing import Any

_GR_MODEL_FAMILIES: dict[str, dict[str, Any]] = {}


def register_model_family(
    model_type: str,
    gpu_worker: str,
    npu_worker: str,
    defaults_fn: str | None = None,
) -> None:
    """Register a model family in the global family table."""
    _GR_MODEL_FAMILIES[model_type] = {
        "gpu_worker": gpu_worker,
        "npu_worker": npu_worker,
        "defaults_fn": defaults_fn,
    }


def get_registered_families() -> set[str]:
    """Return the set of currently registered model family names."""
    return set(_GR_MODEL_FAMILIES.keys())


def get_worker_cls_path(model_type: str, is_npu: bool) -> str | None:
    """Return the dotted worker class path for the given family and platform."""
    family = _GR_MODEL_FAMILIES.get(model_type)
    if family is None:
        return None
    return family.get("npu_worker" if is_npu else "gpu_worker")


def get_family_info(model_type: str) -> dict[str, Any] | None:
    """Return the full family info dict, or ``None`` if not registered."""
    return _GR_MODEL_FAMILIES.get(model_type)


def register_hstu_family() -> None:
    """Idempotent registration of the HSTU model family."""
    if "hstu" in _GR_MODEL_FAMILIES:
        return
    register_model_family(
        model_type="hstu",
        gpu_worker="vllm_gr.worker.gr_worker.GRWorker",
        npu_worker="vllm_gr.worker.gr_npu_worker.GRNPUWorker",
    )
