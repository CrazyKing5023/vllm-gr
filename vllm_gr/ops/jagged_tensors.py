# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
Jagged Tensor Operator Entry Point

Jagged Tensor is the core data structure for handling variable-length sequences.
Currently uses PyTorch implementation.
"""

from typing import Optional, Tuple

import torch

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_split_2D_jagged,
)


def concat_2D_jagged(
    max_seq_len: int,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Concatenate two 2D Jagged Tensors
    
    Used for merging user history sequence and candidate item sequence.
    """
    return pytorch_concat_2D_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left,
        offsets_right=offsets_right,
    )


def split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int] = None,
    total_len_right: Optional[int] = None,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split 2D Jagged Tensor into two parts
    
    Used for separating user embeddings and candidate embeddings from merged sequence.
    """
    return pytorch_split_2D_jagged(
        max_seq_len=max_seq_len,
        values=values,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left,
        offsets_right=offsets_right,
    )
