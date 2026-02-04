# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""PyTorch implementation of jagged tensor operations"""

from typing import Optional, Tuple

import torch


def pytorch_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Concatenate two 2D jagged tensors along sequence dimension.
    
    Args:
        values_left: Left values (L_left, D)
        values_right: Right values (L_right, D)
        max_len_left: Max length of left sequences
        max_len_right: Max length of right sequences (dense case)
        offsets_left: Offsets for left (B+1,)
        offsets_right: Offsets for right (B+1,) or None for dense
    
    Returns:
        Concatenated tensor (L_total, D)
    """
    D = values_left.shape[1]
    device = values_left.device
    dtype = values_left.dtype
    
    # Determine batch size
    if offsets_left is not None:
        B = offsets_left.size(0) - 1
    elif offsets_right is not None:
        B = offsets_right.size(0) - 1
    else:
        raise ValueError("At least one of offsets_left or offsets_right must be provided")
    
    results = []
    
    for i in range(B):
        # Get left values
        if offsets_left is not None:
            left_start = int(offsets_left[i].item())
            left_end = int(offsets_left[i + 1].item())
            left_vals = values_left[left_start:left_end]
        else:
            # Dense case: max_len_left per batch
            left_start = i * max_len_left
            left_end = (i + 1) * max_len_left
            left_vals = values_left[left_start:left_end]
        
        # Get right values
        if offsets_right is not None:
            right_start = int(offsets_right[i].item())
            right_end = int(offsets_right[i + 1].item())
            right_vals = values_right[right_start:right_end]
        else:
            # Dense case: max_len_right per batch
            right_start = i * max_len_right
            right_end = (i + 1) * max_len_right
            right_vals = values_right[right_start:right_end]
        
        results.append(left_vals)
        results.append(right_vals)
    
    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.empty(0, D, dtype=dtype, device=device)


def pytorch_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split 2D jagged tensor into two parts.
    
    Args:
        max_seq_len: Maximum sequence length
        values: Input values (L, D)
        max_len_left: Max length of left sequences
        max_len_right: Max length of right sequences
        offsets_left: Offsets for left output (B+1,)
        offsets_right: Offsets for right output (B+1,)
    
    Returns:
        (left_values, right_values)
    """
    D = values.shape[1]
    device = values.device
    dtype = values.dtype
    
    # Determine batch size
    if offsets_left is not None:
        B = offsets_left.size(0) - 1
    elif offsets_right is not None:
        B = offsets_right.size(0) - 1
    else:
        raise ValueError("At least one offset must be provided")
    
    left_results = []
    right_results = []
    
    # Compute combined offsets (assume values are concatenated left+right)
    combined_offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
    for i in range(B):
        left_len = 0
        right_len = 0
        
        if offsets_left is not None:
            left_len = int(offsets_left[i + 1].item()) - int(offsets_left[i].item())
        elif max_len_left is not None:
            left_len = max_len_left
        
        if offsets_right is not None:
            right_len = int(offsets_right[i + 1].item()) - int(offsets_right[i].item())
        elif max_len_right is not None:
            right_len = max_len_right
        
        combined_offsets[i + 1] = combined_offsets[i] + left_len + right_len
    
    for i in range(B):
        combined_start = int(combined_offsets[i].item())
        combined_end = int(combined_offsets[i + 1].item())
        
        if offsets_left is not None:
            left_len = int(offsets_left[i + 1].item()) - int(offsets_left[i].item())
        elif max_len_left is not None:
            left_len = max_len_left
        else:
            left_len = 0
        
        left_results.append(values[combined_start:combined_start + left_len])
        right_results.append(values[combined_start + left_len:combined_end])
    
    left_values = torch.cat(left_results, dim=0) if left_results else torch.empty(0, D, dtype=dtype, device=device)
    right_values = torch.cat(right_results, dim=0) if right_results else torch.empty(0, D, dtype=dtype, device=device)
    
    return left_values, right_values


def pytorch_hstu_concat_l2_embeddings(
    contextual_seq_len: int,
    max_prefix_len: int,
    prefix_x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    max_l2_len: int,
    l2_x: torch.Tensor,
    l2_offsets: torch.Tensor,
) -> torch.Tensor:
    """Concatenate L2 embeddings with prefix embeddings."""
    return pytorch_concat_2D_jagged(
        values_left=prefix_x,
        values_right=l2_x,
        max_len_left=max_prefix_len,
        max_len_right=max_l2_len,
        offsets_left=prefix_offsets,
        offsets_right=l2_offsets,
    )


def pytorch_hstu_split_l2_embeddings(
    max_seq_len: int,
    x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split embeddings into prefix and L2 parts."""
    return pytorch_split_2D_jagged(
        max_seq_len=max_seq_len,
        values=x,
        offsets_left=prefix_offsets,
        offsets_right=l2_offsets,
    )

