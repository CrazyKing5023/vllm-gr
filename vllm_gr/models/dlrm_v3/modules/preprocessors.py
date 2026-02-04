# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""Input preprocessors for HSTU model"""

import abc
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch

from vllm_gr.models.dlrm_v3.common import (
    HammerModule,
    init_mlp_weights_optional_bias,
    jagged_to_padded_dense,
)
from vllm_gr.ops.jagged_tensors import concat_2D_jagged
from vllm_gr.ops.layer_norm import LayerNorm, SwishLayerNorm


def complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Compute complete cumulative sum (starting with 0)."""
    try:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except (RuntimeError, AttributeError):
        # Pure PyTorch fallback
        cumsum = torch.zeros(lengths.size(0) + 1, dtype=lengths.dtype, device=lengths.device)
        cumsum[1:] = torch.cumsum(lengths, dim=0)
        return cumsum


class InputPreprocessor(HammerModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        pass

    def interleave_targets(self) -> bool:
        return False


class ContextualPreprocessor(InputPreprocessor):
    """Contextual preprocessor for HSTU model."""
    
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        action_embedding_dim: int = 8,
        action_feature_name: str = "",
        action_weights: Optional[List[int]] = None,
        additional_embedding_features: List[str] = [],
        action_embedding_init_std: float = 0.1,
        is_inference: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        
        self._output_embedding_dim: int = output_embedding_dim
        self._input_embedding_dim: int = input_embedding_dim
        self._hidden_dim: int = hidden_dim
        self._contextual_feature_to_max_length: Dict[str, int] = contextual_feature_to_max_length
        self._max_contextual_seq_len: int = sum(contextual_feature_to_max_length.values())
        self._contextual_feature_to_min_uih_length: Dict[str, int] = contextual_feature_to_min_uih_length
        
        # Contextual linear layers
        if self._max_contextual_seq_len > 0:
            std = 1.0 * sqrt(2.0 / float(input_embedding_dim + self._output_embedding_dim))
            self._batched_contextual_linear_weights: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty((
                    self._max_contextual_seq_len,
                    input_embedding_dim,
                    self._output_embedding_dim,
                )).normal_(0.0, std)
            )
            self._batched_contextual_linear_bias: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty((self._max_contextual_seq_len, self._output_embedding_dim)).fill_(0.0)
            )
        
        # Content embedding MLP
        self._content_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._input_embedding_dim, out_features=self._hidden_dim),
            SwishLayerNorm(self._hidden_dim),
            torch.nn.Linear(in_features=self._hidden_dim, out_features=self._output_embedding_dim),
            LayerNorm(self._output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)
        
        # Additional embedding features
        self._additional_embedding_features: List[str] = additional_embedding_features
        if additional_embedding_features:
            self._additional_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self._input_embedding_dim * len(additional_embedding_features),
                    out_features=self._hidden_dim,
                ),
                SwishLayerNorm(self._hidden_dim),
                torch.nn.Linear(in_features=self._hidden_dim, out_features=self._output_embedding_dim),
                LayerNorm(self._output_embedding_dim),
            ).apply(init_mlp_weights_optional_bias)
        
        # Action embedding (current simplified version, not using action encoder)
        self._action_weights: Optional[List[int]] = action_weights

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        # Apply content embedding MLP
        output_seq_embeddings = self._content_embedding_mlp(seq_embeddings)
        
        # Add additional embeddings if present
        if len(self._additional_embedding_features) > 0 and hasattr(self, '_additional_embedding_mlp'):
            additional_embeddings = torch.cat(
                [seq_payloads[feature] for feature in self._additional_embedding_features if feature in seq_payloads],
                dim=1,
            )
            if additional_embeddings.size(0) > 0:
                output_seq_embeddings = output_seq_embeddings + self._additional_embedding_mlp(additional_embeddings)
        
        max_seq_len = max_uih_len + max_targets
        target_offsets = complete_cumsum(num_targets)
        seq_offsets = complete_cumsum(seq_lengths)
        
        output_max_seq_len = max_seq_len
        output_total_uih_len = total_uih_len
        output_total_targets = total_targets
        output_seq_lengths = seq_lengths
        output_num_targets = num_targets
        output_seq_timestamps = seq_timestamps
        output_seq_offsets = complete_cumsum(output_seq_lengths)
        
        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            seq_payloads,
        )
