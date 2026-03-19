# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Device detection and backend selection helpers."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceType:
    """Canonical device backend names used by vllm-gr."""

    CUDA = "cuda"
    NPU = "npu"
    CPU = "cpu"
    UNKNOWN = "unknown"


def get_device_backend(device: torch.device) -> str:
    """Return the canonical backend name for a torch device."""
    device_type = getattr(device, "type", None)
    if device_type is None:
        return DeviceType.UNKNOWN

    device_type = str(device_type).lower()
    if device_type in (DeviceType.CUDA, DeviceType.NPU, DeviceType.CPU):
        return device_type
    return DeviceType.UNKNOWN


def is_npu_available() -> bool:
    """Return whether torch_npu is installed and the runtime is available."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """Return whether CUDA is available."""
    try:
        return torch.cuda.is_available()
    except AttributeError:
        return False


def get_default_device() -> torch.device:
    """Prefer NPU, then CUDA, then CPU."""
    if is_npu_available():
        return torch.device(DeviceType.NPU)
    if is_cuda_available():
        return torch.device(DeviceType.CUDA)
    return torch.device(DeviceType.CPU)


def check_torchrec_available() -> bool:
    """Return whether the torchrec or torchrec-npu runtime is available."""
    try:
        import torchrec  # noqa: F401
        return True
    except ImportError:
        return False


def check_recsdk_available() -> bool:
    """Backward-compatible alias for the torchrec runtime availability check."""
    return check_torchrec_available()


def check_fbgemm_available() -> bool:
    """Return whether the required fbgemm custom ops are available."""
    try:
        torch.ops.fbgemm.asynchronous_complete_cumsum
        torch.ops.fbgemm.jagged_to_padded_dense
        torch.ops.fbgemm.dense_to_jagged
        return True
    except (RuntimeError, AttributeError):
        return False


class DeviceAdapter:
    """Small facade around backend detection and optional operator stacks."""

    def __init__(self) -> None:
        self._device_backend = self._detect_device_backend()
        self._torchrec_available = check_torchrec_available()
        self._fbgemm_available = check_fbgemm_available()

        logger.debug(
            "DeviceAdapter backend=%s torchrec=%s fbgemm=%s",
            self._device_backend,
            self._torchrec_available,
            self._fbgemm_available,
        )

    def _detect_device_backend(self) -> str:
        if is_npu_available():
            return DeviceType.NPU
        if is_cuda_available():
            return DeviceType.CUDA
        return DeviceType.CPU

    @property
    def device_backend(self) -> str:
        return self._device_backend

    @property
    def is_npu(self) -> bool:
        return self._device_backend == DeviceType.NPU

    @property
    def is_cuda(self) -> bool:
        return self._device_backend == DeviceType.CUDA

    @property
    def is_cpu(self) -> bool:
        return self._device_backend == DeviceType.CPU

    def should_use_torchrec(self, device: Optional[torch.device] = None) -> bool:
        """Return whether a torchrec-backed NPU environment is available."""
        target = self.device_backend if device is None else get_device_backend(device)
        return target == DeviceType.NPU and self._torchrec_available

    def should_use_recsdk(self, device: Optional[torch.device] = None) -> bool:
        """Backward-compatible alias for the torchrec-backed NPU check."""
        return self.should_use_torchrec(device)

    def should_use_fbgemm(self, device: Optional[torch.device] = None) -> bool:
        """Return whether fbgemm-backed execution is usable for a device."""
        target = self.device_backend if device is None else get_device_backend(device)
        if not self._fbgemm_available:
            return False

        if target == DeviceType.CUDA:
            return True

        if target == DeviceType.NPU:
            return self._torchrec_available

        return False


_global_adapter: Optional[DeviceAdapter] = None


def get_device_adapter() -> DeviceAdapter:
    """Return a process-wide adapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = DeviceAdapter()
    return _global_adapter


def reset_device_adapter() -> None:
    """Reset the process-wide adapter. Primarily used in tests."""
    global _global_adapter
    _global_adapter = None


def get_device_type(device: torch.device) -> str:
    return get_device_backend(device)


def is_npu_device(device: torch.device) -> bool:
    return get_device_backend(device) == DeviceType.NPU


def is_cuda_device(device: torch.device) -> bool:
    return get_device_backend(device) == DeviceType.CUDA
