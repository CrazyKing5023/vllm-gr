# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.pytorch.pt_position import (
    pytorch_add_timestamp_positional_embeddings,
)


def add_timestamp_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    assert time_bucket_fn in ["sqrt", "log"]
    seq_embeddings = seq_embeddings * alpha
    if kernel == HammerKernel.PYTORCH:
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    # vllm-GR currently forces HammerKernel.PYTORCH globally.
    return pytorch_add_timestamp_positional_embeddings(
        seq_embeddings=seq_embeddings,
        seq_offsets=seq_offsets,
        pos_embeddings=position_embeddings_weight,
        ts_embeddings=timestamp_embeddings_weight,
        timestamps=timestamps,
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
        time_bucket_fn=time_bucket_fn,
    )
