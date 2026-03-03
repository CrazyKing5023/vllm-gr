# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Input data types for vllm-GR.
"""

from vllm_gr.inputs.data import (
    GRPrompt,
    is_gr_prompt,
)

__all__ = [
    "GRPrompt",
    "is_gr_prompt",
]
