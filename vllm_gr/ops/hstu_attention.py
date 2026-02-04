# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
HSTU 注意力算子入口

当前版本使用 PyTorch 实现，后续可扩展 Triton/CUDA 优化版本。
"""

from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerKernel, switch_to_contiguous_if_needed
from vllm_gr.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    """
    HSTU 多头注意力
    
    当前使用 PyTorch 实现，kernel 参数保留用于未来扩展。
    """
    # 使用 PyTorch 实现
    return pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=dropout_pr,
        training=training,
        num_targets=num_targets,
        attn_scale=attn_scale,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )


def delta_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    """
    增量 HSTU 注意力（用于 KV 缓存推理）
    
    当前使用 PyTorch 实现。
    """
    return pytorch_cached_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )

