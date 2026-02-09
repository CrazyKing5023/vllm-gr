# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""HSTU Positional Encoder"""

from math import sqrt
from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerModule


class HSTUPositionalEncoder(HammerModule):
    """
    Positional encoder for HSTU model.
    Adds position and timestamp embeddings to sequence embeddings.
    """
    
    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        contextual_seq_len: int,
        is_inference: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._embedding_dim: int = embedding_dim
        self._contextual_seq_len: int = contextual_seq_len
        
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                -sqrt(1.0 / num_time_buckets),
                sqrt(1.0 / num_time_buckets),
            ),
        )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Add positional encoding to sequence embeddings
        
        Includes both position embeddings and timestamp embeddings.
        """
        L = seq_embeddings.size(0)
        device = seq_embeddings.device
        dtype = seq_embeddings.dtype
        
        # Create position indices
        positions = torch.zeros(L, dtype=torch.long, device=device)
        B = seq_offsets.size(0) - 1
        
        for i in range(B):
            start = int(seq_offsets[i].item())
            end = int(seq_offsets[i + 1].item())
            length = end - start
            positions[start:end] = torch.arange(length, device=device)
        
        # Clip positions to valid range
        positions = positions.clamp(0, self._position_embeddings_weight.size(0) - 1)
        
        # Add position embeddings
        pos_emb = self._position_embeddings_weight[positions]
        
        # Compute time bucket indices from timestamps
        if seq_timestamps is not None and seq_timestamps.numel() > 0:
            # Simple time bucket computation (using direct mapping)
            time_buckets = (seq_timestamps.abs() % (self._timestamp_embeddings_weight.size(0) - 1)).long()
            time_emb = self._timestamp_embeddings_weight[time_buckets]
            
            # Combine: scale by sqrt(dim) and add
            seq_embeddings = seq_embeddings * (self._embedding_dim ** 0.5) + pos_emb.to(dtype) + time_emb.to(dtype)
        else:
            seq_embeddings = seq_embeddings * (self._embedding_dim ** 0.5) + pos_emb.to(dtype)
        
        return seq_embeddings
