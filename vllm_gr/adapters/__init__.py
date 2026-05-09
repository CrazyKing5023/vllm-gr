# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""GR model adapter framework.

Provides ``GRModelAdapter`` base class, capability protocols, and the
two-layer adapter registry for model-specific recommendation inference
pipelines.
"""

from vllm_gr.adapters.base import GRModelAdapter
from vllm_gr.adapters.registry import (
    apply_family_defaults,
    create_adapter,
    ensure_adapter_class_registered,
    get_adapter_class,
    get_family_info,
    get_payload_type,
    get_registered_families,
    get_worker_cls_path,
    register_adapter_class,
    register_family,
    resolve_model_type,
)

__all__ = [
    "GRModelAdapter",
    # Registry
    "apply_family_defaults",
    "create_adapter",
    "ensure_adapter_class_registered",
    "get_adapter_class",
    "get_family_info",
    "get_payload_type",
    "get_registered_families",
    "get_worker_cls_path",
    "register_adapter_class",
    "register_family",
    "resolve_model_type",
]
