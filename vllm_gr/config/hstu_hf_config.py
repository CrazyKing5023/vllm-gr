# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Compatibility shell for HSTU config and vLLM registration helpers."""

from vllm_gr.config.hstu_config import HSTUConfig
from vllm_gr.config.hstu_vllm_registration import (
    HSTUModelArchConfigConvertor,
    ensure_hstu_config_registered,
    is_hstu_config_registered,
)

__all__ = [
    "HSTUConfig",
    "HSTUModelArchConfigConvertor",
    "ensure_hstu_config_registered",
    "is_hstu_config_registered",
]
