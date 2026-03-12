# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
Layer Normalization Operators

Contains standard LayerNorm and SwishLayerNorm (LayerNorm + SiLU activation) implementations.
"""

import torch
from typing import List

from vllm_gr.models.dlrm_v3.common import HammerKernel, HammerModule

from vllm_gr.ops.pytorch.pt_hstu_layer_norm import (
    pytorch_swish_layer_norm,
    pytorch_layer_norm,
)


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    LayerNorm function interface supporting multiple kernel implementations.

    Args:
        input: Input tensor
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter, default 1e-5
        kernel: Computation kernel type, currently only supports PYTORCH

    Returns:
        Normalized tensor

    Raises:
        ValueError: When using unsupported kernel type
    """
    if kernel == HammerKernel.PYTORCH:
        return pytorch_layer_norm(
            input,
            [input.shape[-1]],
            weight,
            bias,
            eps,
        )
    else:
        raise ValueError("Currently, kernel only supports HammerKernel.PYTORCH.")


class LayerNorm(HammerModule):
    """
    LayerNorm module class.

    Implements standard layer normalization operation for stabilizing neural network training.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
        calculate_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize LayerNorm module.

        Args:
            dim: Feature dimension
            eps: Numerical stability parameter, default 1e-5
            is_inference: Whether in inference mode, default False
            calculate_dtype: Data type for calculation, default torch.float32
        """
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self._eps = eps
        self.calculate_dtype = calculate_dtype
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape, dtype=self.calculate_dtype))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape, dtype=self.calculate_dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Input tensor

        Returns:
            Normalized tensor
        """
        input_dtype = input.dtype
        input = input.to(self.calculate_dtype)
        result = layer_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )
        return result.to(input_dtype)


def swish_layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Swish + LayerNorm function interface supporting multiple kernel implementations.

    Args:
        input: Input tensor
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter, default 1e-5
        kernel: Computation kernel type, currently only supports PYTORCH

    Returns:
        Tensor after LayerNorm and Swish activation

    Raises:
        ValueError: When using unsupported kernel type
    """
    if kernel == HammerKernel.PYTORCH:
        return pytorch_swish_layer_norm(
            input,
            [input.shape[-1]],
            weight,
            bias,
            eps,
        )
    else:
        raise ValueError("Currently, kernel only supports HammerKernel.PYTORCH.")


class SwishLayerNorm(HammerModule):
    """
    SwishLayerNorm module class.

    Implements combined LayerNorm followed by Swish (SiLU) activation operation.
    Swish activation function is defined as: input * sigmoid(input)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
        calculate_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize SwishLayerNorm module.

        Args:
            dim: Feature dimension
            eps: Numerical stability parameter, default 1e-5
            is_inference: Whether in inference mode, default False
            calculate_dtype: Data type for calculation, default torch.float32
        """
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self.calculate_dtype = calculate_dtype
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape, dtype=self.calculate_dtype))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape, dtype=self.calculate_dtype))
        self._eps = eps

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Input tensor

        Returns:
            Tensor after LayerNorm and Swish activation
        """
        input_dtype = input.dtype
        input = input.to(self.calculate_dtype)
        result = swish_layer_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )
        return result.to(input_dtype)
