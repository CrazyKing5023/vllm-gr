# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
DLRM_v3 Model Common Utilities

Contains HammerModule base class, kernel selection, and helper functions.
"""

import abc
import copy
from enum import Enum, unique
from typing import Any, List, Optional, Tuple

import torch


@unique
class HammerKernel(Enum):
    """Kernel implementation options"""
    TRITON = "TRITON"
    TLX = "TLX"
    PYTORCH = "PYTORCH"
    CUDA = "CUDA"
    TRITON_CC = "TRITON_CC"


class HammerModule(torch.nn.Module, abc.ABC):
    """
    HSTU Module Base Class
    
    Provides kernel selection and inference mode control.
    Current version uses PyTorch kernel implementation.
    """
    _is_inference: bool = False
    _use_triton_cc: bool = False  # Currently disabled
    _training_dtype: torch.dtype = torch.float32
    _hammer_kernel: Optional[HammerKernel] = HammerKernel.PYTORCH  # Currently uses PyTorch

    def __init__(
        self,
        is_inference: bool,
        training_dtype: torch.dtype = torch.float32,
        use_triton_cc: bool = False,  # Ignored in current version
        hammer_kernel: Optional[HammerKernel] = None,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._training_dtype = training_dtype
        # Current version uses PyTorch kernel
        self._hammer_kernel = HammerKernel.PYTORCH
        self._use_triton_cc = False

    def hammer_kernel(self) -> HammerKernel:
        """Return current kernel type"""
        return HammerKernel.PYTORCH

    def recursive_setattr(self, name: str, value: Any) -> None:
        for _, module in self.named_modules():
            if hasattr(module, name):
                setattr(module, name, value)

    def set_use_triton_cc(self, use_triton_cc: bool) -> None:
        # Not supported in current version, interface reserved for future extension
        pass

    def set_is_inference(self, is_inference: bool) -> None:
        self._is_inference = is_inference
        self.recursive_setattr("_is_inference", is_inference)

    def set_training_dtype(self, training_dtype: torch.dtype) -> None:
        self._training_dtype = training_dtype
        self.recursive_setattr("_training_dtype", training_dtype)

    def set_hammer_kernel(self, hammer_kernel: HammerKernel) -> None:
        # Not supported in current version, interface reserved for future extension
        pass

    @property
    def is_inference(self) -> bool:
        return self._is_inference

    @property
    def is_eval(self) -> bool:
        return (not self._is_inference) and (not self.training)

    @property
    def is_train(self) -> bool:
        return (not self._is_inference) and self.training


# Global settings
STATIC_MAX_SEQ_LENS: List[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = True


def set_static_max_seq_lens(max_seq_lens: List[int]) -> None:
    global STATIC_MAX_SEQ_LENS
    STATIC_MAX_SEQ_LENS = copy.deepcopy(max_seq_lens)
    STATIC_MAX_SEQ_LENS.sort()


def set_use_runtime_max_seq_len(use_runtime_max_seq_len: bool) -> None:
    global USE_RUNTIME_MAX_SEQ_LEN
    USE_RUNTIME_MAX_SEQ_LEN = use_runtime_max_seq_len


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def prev_power_of_2(x: int) -> int:
    out = next_power_of_2(x)
    return out // 2 if out > x else out


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    global USE_RUNTIME_MAX_SEQ_LEN
    
    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def fx_infer_max_len(lengths: torch.Tensor) -> int:
    max_len = int(lengths.max().item())
    return max_len


@torch.fx.wrap
def fx_mark_length_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)


@torch.fx.wrap
def fx_torch_ones(
    shape: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=dtype)


@torch.fx.wrap
def fx_torch_zeros(shape: List[int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, device=device)


def init_mlp_weights_optional_bias(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# GPU availability checks
nv_gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
nv_gpu_available: bool = not nv_gpu_unavailable[0]
gpu_available: bool = nv_gpu_available


def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float,
) -> torch.Tensor:
    """
    Convert Jagged Tensor to padded dense tensor
    
    Prefers fbgemm, falls back to PyTorch implementation if unavailable.
    """
    try:
        return torch.ops.fbgemm.jagged_to_padded_dense(
            values=values,
            offsets=offsets,
            max_lengths=max_lengths,
            padding_value=padding_value,
        )
    except (RuntimeError, AttributeError):
        # Pure PyTorch fallback
        return _pytorch_jagged_to_padded_dense(
            values=values,
            offsets=offsets[0] if offsets else torch.tensor(
                [0, values.size(0)], dtype=torch.long, device=values.device
            ),
            max_length=max_lengths[0] if max_lengths else values.size(0),
            padding_value=padding_value,
        )


def _pytorch_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float,
) -> torch.Tensor:
    """Pure PyTorch implementation of jagged_to_padded_dense"""
    B = offsets.size(0) - 1
    D = values.size(-1) if values.dim() > 1 else 1
    
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    
    output = torch.full((B, max_length, D), padding_value, 
                        dtype=values.dtype, device=values.device)
    
    for i in range(B):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = min(end - start, max_length)
        if length > 0:
            output[i, :length] = values[start:start + length]
    
    if D == 1:
        output = output.squeeze(-1)
    
    return output


def dense_to_jagged(
    dense: torch.Tensor,
    x_offsets: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert padded dense tensor to Jagged Tensor
    
    Prefers fbgemm, falls back to PyTorch implementation if unavailable.
    """
    try:
        return torch.ops.fbgemm.dense_to_jagged(
            dense=dense,
            x_offsets=x_offsets,
        )[0]
    except (RuntimeError, AttributeError):
        # Pure PyTorch fallback
        return _pytorch_dense_to_jagged(
            dense=dense,
            offsets=x_offsets[0] if x_offsets else torch.tensor(
                [0, dense.size(1)], dtype=torch.long, device=dense.device
            ),
        )


def _pytorch_dense_to_jagged(
    dense: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch implementation of dense_to_jagged"""
    B = dense.size(0)
    results = []
    
    for i in range(B):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = end - start
        if length > 0:
            results.append(dense[i, :length])
    
    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.empty(0, *dense.shape[2:], dtype=dense.dtype, device=dense.device)
