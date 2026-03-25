# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""STU (Sequential Transducer Unit) module for HSTU model.

This fork keeps the batch scoring forward path used by inference, but does not
currently restore the full upstream incremental KV-cache update helpers. The
`cached_forward()` interface is left in place so future work can add true
incremental support without changing call sites.
"""

import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function

from vllm_gr.models.dlrm_v3.common import HammerModule, fx_unwrap_optional_tensor
from vllm_gr.ops.hstu_attention import delta_hstu_mha
from vllm_gr.ops.hstu_compute import hstu_compute_output, hstu_compute_uqvk, hstu_preprocess_and_attention
from vllm_gr.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged


class STU(HammerModule, abc.ABC):
    """Abstract base class for STU modules.

    In this inference-focused fork, subclasses are expected to support the
    batch forward path first. Incremental cache semantics remain intentionally
    limited compared with the training/full-runtime upstream implementation.
    """
    
    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Incremental forward is intentionally not implemented in this fork."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    """Configuration for STU layer."""
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0


class STULayer(STU):
    """Single STU layer implementation."""
    
    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        config: STULayerConfig,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self.reset_kv_cache()
        
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim ** 0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len

        # U, V, Q, K projection weights
        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((
                self._embedding_dim,
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            )),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads),
        )
        
        # Input layer norm
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        
        # Output projection
        # When concat_u=True and concat_x=True in pytorch_norm_mul_dropout:
        #   y = torch.cat([u, x, y], dim=1)  where x is actually attn
        # u dim: num_heads * hidden_dim
        # x (attn) dim: num_heads * hidden_dim  
        # y (u * normalized(attn)) dim: num_heads * hidden_dim
        # Total: 3 * num_heads * hidden_dim
        # output_weight shape: (y_dim, embedding_dim) for torch.addmm
        self._output_weight = torch.nn.Parameter(
            torch.empty((self._hidden_dim * self._num_heads * 3, self._embedding_dim)),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        
        # Output layer norm
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )

    def reset_kv_cache(self) -> None:
        """Reset KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of STU layer."""
        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=self.hammer_kernel(),
            )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )


class STUStack(STU):
    """Stack of multiple STU layers."""
    
    def __init__(
        self,
        stu_list: List[STU],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all STU layers."""
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return x
