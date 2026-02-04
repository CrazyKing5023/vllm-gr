# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
Layer Normalization 算子

包含标准 LayerNorm 和 SwishLayerNorm（LayerNorm + SiLU 激活）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_gr.models.dlrm_v3.common import HammerKernel


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    应用 Layer Normalization
    
    使用 PyTorch F.layer_norm 实现。
    """
    return F.layer_norm(x, [x.shape[-1]], weight, bias, eps)


class LayerNorm(nn.Module):
    """Standard Layer Normalization module."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class SwishLayerNorm(nn.Module):
    """Layer Normalization followed by Swish activation."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.layer_norm(x))

