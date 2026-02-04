# Copyright (c) vllm-GR Authors.
# Original code from Meta Platforms, Inc. (generative-recommenders)
# Licensed under the Apache License, Version 2.0

"""
HSTU 模型算子包

包含注意力、计算、Jagged Tensor 操作和归一化等算子。
当前提供 PyTorch 实现，可扩展 Triton/CUDA 优化版本。
"""

from vllm_gr.ops.hstu_attention import hstu_mha, delta_hstu_mha
from vllm_gr.ops.hstu_compute import (
    hstu_compute_uqvk,
    hstu_compute_output,
    hstu_preprocess_and_attention,
)
from vllm_gr.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from vllm_gr.ops.layer_norm import LayerNorm, SwishLayerNorm, layer_norm

__all__ = [
    "hstu_mha",
    "delta_hstu_mha",
    "hstu_compute_uqvk",
    "hstu_compute_output",
    "hstu_preprocess_and_attention",
    "concat_2D_jagged",
    "split_2D_jagged",
    "LayerNorm",
    "SwishLayerNorm",
    "layer_norm",
]

