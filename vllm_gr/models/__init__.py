# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Models package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["DlrmHSTU", "DlrmHSTUConfig"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("vllm_gr.models.dlrm_v3.modules.dlrm_hstu")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
