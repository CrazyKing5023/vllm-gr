# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

import torch
from typing import List

def pytorch_swish_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    PyTorch native Swish + LayerNorm implementation.

    Swish activation function: input * sigmoid(input)
    Performs LayerNorm first, then applies Swish activation.

    Args:
        input: Input tensor
        normalized_shape: Normalized shape
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter

    Returns:
        Tensor after LayerNorm and Swish activation
    """
    return (
        input
        * torch.sigmoid(
            torch.nn.functional.layer_norm(
                input,
                normalized_shape,
                weight,
                bias,
                eps,
            )
        )
    )


def pytorch_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    PyTorch native LayerNorm implementation.

    Args:
        input: Input tensor
        normalized_shape: Normalized shape
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter

    Returns:
        Normalized tensor
    """
    return torch.nn.functional.layer_norm(
        input,
        normalized_shape,
        weight,
        bias,
        eps,
    )
