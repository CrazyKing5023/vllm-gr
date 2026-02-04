# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""PyTorch implementation of HSTU linear operations"""

import torch
import torch.nn.functional as F


def pytorch_hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_u: bool,
    concat_x: bool,
    group_norm: bool,
    num_heads: int,
    linear_dim: int,
) -> torch.Tensor:
    """
    Compute HSTU output with layer normalization and dropout.
    
    Args:
        attn: Attention output (L, H*D)
        u: Gate values (L, H*D)
        x: Original input (L, E)
        norm_weight, norm_bias: Normalization parameters
        output_weight: Output projection weight
        eps: Normalization epsilon
        dropout_ratio: Dropout ratio
        training: Training mode
        concat_u: Concatenate gate values
        concat_x: Concatenate original input
        group_norm: Use group normalization
        num_heads: Number of attention heads
        linear_dim: Linear dimension
    
    Returns:
        Output tensor (L, E)
    """
    # Apply layer/group normalization
    if group_norm:
        # Reshape for group norm: (L, num_heads, linear_dim)
        attn_reshaped = attn.view(-1, num_heads, linear_dim)
        normalized = F.layer_norm(
            attn_reshaped, 
            [linear_dim], 
            norm_weight, 
            norm_bias, 
            eps
        ).view(-1, num_heads * linear_dim)
    else:
        normalized = F.layer_norm(
            attn, 
            [num_heads * linear_dim], 
            norm_weight, 
            norm_bias, 
            eps
        )
    
    # Apply gate
    y = normalized * u
    
    # Apply dropout
    if training and dropout_ratio > 0:
        y = F.dropout(y, p=dropout_ratio, training=True)
    
    # Concatenate u and x if needed
    if concat_u:
        y = torch.cat([y, u], dim=-1)
    if concat_x:
        y = torch.cat([y, x], dim=-1)
    
    # Output projection
    output = F.linear(y, output_weight)
    
    # Residual connection
    output = output + x
    
    return output

