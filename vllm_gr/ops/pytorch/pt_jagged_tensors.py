# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""PyTorch implementation of jagged tensor operations"""

from typing import Optional, Tuple

import torch


def _pytorch_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float,
) -> torch.Tensor:
    batch_size = offsets.size(0) - 1
    output = torch.full(
        (batch_size, max_length, values.size(-1)),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )

    for idx in range(batch_size):
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        length = min(end - start, max_length)
        if length > 0:
            output[idx, :length] = values[start:start + length]

    return output


def _concat_2D_jagged_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = max_len_left + max_len_right
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    padded_left = _pytorch_jagged_to_padded_dense(
        values=values_left,
        offsets=offsets_left,
        max_length=max_len_left,
        padding_value=0.0,
    )
    padded_right = _pytorch_jagged_to_padded_dense(
        values=values_right,
        offsets=offsets_right,
        max_length=max_len_right,
        padding_value=0.0,
    )
    concatted_dense = torch.cat([padded_left, padded_right], dim=1)
    mask = torch.arange(max_seq_len, device=offsets_left.device).view(1, -1)
    mask = torch.logical_or(
        mask < lengths_left.view(-1, 1),
        torch.logical_and(
            mask >= max_len_left,
            mask < max_len_left + lengths_right.view(-1, 1),
        ),
    )
    return concatted_dense.flatten(0, 1)[mask.view(-1), :]


def pytorch_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if offsets_left is None:
        assert max_len_left is not None
        batch_size = values_left.shape[0] // max_len_left
        offsets_left_non_optional = max_len_left * torch.arange(
            batch_size + 1,
            device=values_left.device,
        )
    else:
        offsets_left_non_optional = offsets_left

    if offsets_right is None:
        assert max_len_right is not None
        batch_size = values_right.shape[0] // max_len_right
        offsets_right_non_optional = max_len_right * torch.arange(
            batch_size + 1,
            device=values_left.device,
        )
    else:
        offsets_right_non_optional = offsets_right

    max_len_left = (
        int(
            (offsets_left_non_optional[1:] - offsets_left_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_left is None
        else max_len_left
    )
    max_len_right = (
        int(
            (offsets_right_non_optional[1:] - offsets_right_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_right is None
        else max_len_right
    )
    return _concat_2D_jagged_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def _split_2D_jagged_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets_left + offsets_right
    padded_values = _pytorch_jagged_to_padded_dense(
        values=values,
        offsets=offsets,
        max_length=max_seq_len,
        padding_value=0.0,
    ).flatten(0, 1)
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    mask = torch.arange(max_seq_len, device=values.device).view(1, -1)
    mask_left = mask < lengths_left.view(-1, 1)
    mask_right = torch.logical_and(
        mask >= lengths_left.view(-1, 1),
        mask < (lengths_left + lengths_right).view(-1, 1),
    )
    return padded_values[mask_left.view(-1), :], padded_values[mask_right.view(-1), :]


def pytorch_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets_left is None:
        assert max_len_left is not None
        assert offsets_right is not None
        offsets_left_non_optional = max_len_left * torch.arange(
            offsets_right.shape[0],
            device=values.device,
        )
    else:
        offsets_left_non_optional = offsets_left

    if offsets_right is None:
        assert max_len_right is not None
        assert offsets_left is not None
        offsets_right_non_optional = max_len_right * torch.arange(
            offsets_left.shape[0],
            device=values.device,
        )
    else:
        offsets_right_non_optional = offsets_right

    return _split_2D_jagged_jagged(
        max_seq_len=max_seq_len,
        values=values,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def pytorch_hstu_concat_l2_embeddings(
    max_prefix_len: int,
    prefix_x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    max_l2_len: int,
    l2_x: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> torch.Tensor:
    """Restore ``[contextual][prefix][remaining_l2]`` jagged layout."""
    padded_prefix_x = _pytorch_jagged_to_padded_dense(
        values=prefix_x,
        offsets=prefix_offsets,
        max_length=max_prefix_len,
        padding_value=0.0,
    )
    padded_l2_x = _pytorch_jagged_to_padded_dense(
        values=l2_x,
        offsets=l2_offsets,
        max_length=max_l2_len,
        padding_value=0.0,
    )
    padded_x = torch.cat(
        [
            padded_l2_x[:, 0:contextual_seq_len, :],
            padded_prefix_x,
            padded_l2_x[:, contextual_seq_len:, :],
        ],
        dim=1,
    )
    mask = torch.arange(
        max_prefix_len + max_l2_len,
        device=prefix_x.device,
    ).view(1, -1)
    prefix_lengths = prefix_offsets[1:] - prefix_offsets[:-1]
    l2_lengths = l2_offsets[1:] - l2_offsets[:-1]
    mask = torch.logical_or(
        mask < prefix_lengths.view(-1, 1) + contextual_seq_len,
        torch.logical_and(
            mask >= max_prefix_len + contextual_seq_len,
            mask < max_prefix_len + l2_lengths.view(-1, 1),
        ),
    )
    return padded_x.flatten(0, 1)[mask.view(-1), :]


def pytorch_hstu_split_l2_embeddings(
    max_seq_len: int,
    x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``[contextual][prefix][remaining_l2]`` into prefix and L2 views."""
    x_offsets = prefix_offsets + l2_offsets
    x_lengths = x_offsets[1:] - x_offsets[:-1]
    padded_x = _pytorch_jagged_to_padded_dense(
        values=x,
        offsets=x_offsets,
        max_length=max_seq_len,
        padding_value=0.0,
    )
    padded_x = padded_x.flatten(0, 1)
    prefix_lengths = prefix_offsets[1:] - prefix_offsets[:-1]
    mask = torch.arange(max_seq_len, device=x_offsets.device).view(1, -1)
    mask_prefix = torch.logical_and(
        mask >= contextual_seq_len,
        mask < prefix_lengths.view(-1, 1) + contextual_seq_len,
    )
    mask_l2 = torch.logical_or(
        mask < contextual_seq_len,
        torch.logical_and(
            mask >= prefix_lengths.view(-1, 1) + contextual_seq_len,
            mask < x_lengths.view(-1, 1),
        ),
    )
    return padded_x[mask_prefix.view(-1), :], padded_x[mask_l2.view(-1), :]
