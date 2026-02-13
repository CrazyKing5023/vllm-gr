# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""Output postprocessors for HSTU model"""

from abc import abstractmethod
from typing import Dict, List, Tuple

import torch

from vllm_gr.models.dlrm_v3.common import HammerModule, init_mlp_weights_optional_bias


@torch.fx.wrap
def _cast_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != dtype:
        return t.to(dtype)
    return t


class OutputPostprocessor(HammerModule):
    """An abstract class for post-processing user embeddings after HSTU layers."""

    @abstractmethod
    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass


class L2NormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with L2 normalization."""

    def __init__(self, is_inference: bool = False) -> None:
        super().__init__(is_inference=is_inference)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return seq_embeddings / torch.linalg.norm(
            seq_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)


class LayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with layer normalization."""

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._layer_norm(seq_embeddings.to(self._layer_norm.weight.dtype))


@torch.fx.wrap
def _unsqueeze_if_needed(t: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 3:
        return t.unsqueeze(0)
    return t


class TimestampLayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with timestamp-based MLP -> layer norm."""

    def __init__(
        self,
        embedding_dim: int,
        time_duration_features: List[Tuple[int, int]],
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )
        self.register_buffer(
            "_period_units",
            torch.Tensor([f[0] for f in time_duration_features]).view(1, -1),
        )
        self.register_buffer(
            "_units_per_period",
            torch.Tensor([f[1] for f in time_duration_features]).view(1, -1),
        )
        self._time_feature_combiner: torch.nn.Module = torch.nn.Linear(
            embedding_dim + 2 * len(time_duration_features),
            embedding_dim,
        ).apply(init_mlp_weights_optional_bias)

    def _concat_time_features(
        self,
        combined_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure registered buffers are on the same device as input tensors.
        # Buffers created via register_buffer(torch.Tensor([...])) may remain
        # on CPU if they were not covered by the device context during model
        # creation, and vLLM's load_weights only moves nn.Parameters.
        target_device = combined_embeddings.device
        period_units = self._period_units.to(target_device)
        units_per_period = self._units_per_period.to(target_device)

        timestamps = timestamps.unsqueeze(-1)
        period_units = _unsqueeze_if_needed(period_units, combined_embeddings)
        units_per_period = _unsqueeze_if_needed(units_per_period, combined_embeddings)
        
        _units_since_epoch = torch.div(
            timestamps, period_units, rounding_mode="floor"
        )
        _units_elapsed = (
            (torch.remainder(_units_since_epoch, units_per_period) / units_per_period)
            * 2
            * 3.14
        )
        
        _units_elapsed_type: torch.dtype = _units_elapsed.dtype
        _units_elapsed = torch.view_as_real(
            torch.polar(
                _cast_dtype(torch.ones_like(_units_elapsed), torch.float32),
                _cast_dtype(_units_elapsed, torch.float32),
            )
        ).flatten(-2, -1)
        _units_elapsed = _cast_dtype(_units_elapsed, _units_elapsed_type)
        
        # Cast time features to match embedding dtype before concatenation.
        # Timestamps and period_units are always float32 (timestamps are created
        # explicitly as float32, and torch.Tensor([...]) always creates float32).
        # But embeddings may be float16/bfloat16 (vLLM loads models in half
        # precision). Without this cast, torch.cat promotes to float32, causing
        # a dtype mismatch in the downstream nn.Linear (float16 weights).
        _units_elapsed = _cast_dtype(_units_elapsed, combined_embeddings.dtype)
        
        combined_embeddings = torch.cat([combined_embeddings, _units_elapsed], dim=-1)
        return combined_embeddings

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        user_embeddings = self._time_feature_combiner(
            self._concat_time_features(seq_embeddings, timestamps=seq_timestamps)
        )
        return self._layer_norm(user_embeddings)

