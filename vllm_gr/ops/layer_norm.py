# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

"""
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
    ) -> None:
        """
        Initialize LayerNorm module.

        Args:
            dim: Feature dimension
            eps: Numerical stability parameter, default 1e-5
            is_inference: Whether in inference mode, default False
        """
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self._eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Input tensor

        Returns:
            Normalized tensor
        """
        return layer_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )


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
    ) -> None:
        """
        Initialize SwishLayerNorm module.

        Args:
            dim: Feature dimension
            eps: Numerical stability parameter, default 1e-5
            is_inference: Whether in inference mode, default False
        """
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape))
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
        return swish_layer_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )
