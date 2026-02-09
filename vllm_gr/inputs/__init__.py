# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Input data types for vllm-GR.
"""

from vllm_gr.inputs.data import (
    ItemInteraction,
    CandidateItem,
    GRPrompt,
    GRInputs,
    is_gr_prompt,
    gr_inputs,
)

__all__ = [
    "ItemInteraction",
    "CandidateItem",
    "GRPrompt",
    "GRInputs",
    "is_gr_prompt",
    "gr_inputs",
]
