# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Shared runtime helpers for HSTU execution."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def _is_embedding_tensor(name: str) -> bool:
    return "embedding_collection" in name


@dataclass(frozen=True)
class HSTURuntimeDevices:
    """Resolved runtime device policy for the two-phase HSTU execution path."""

    dense_device: torch.device
    embedding_device: torch.device


@dataclass(frozen=True)
class HSTUSparseTransferPlan:
    """Resolved transfer contract from phase-1 sparse outputs to dense phase."""

    source_device: torch.device
    target_device: torch.device
    requires_transfer: bool


def resolve_hstu_runtime_devices(
    module: Optional[torch.nn.Module] = None,
    *,
    explicit_dense_device: Optional[torch.device] = None,
    explicit_embedding_device: Optional[torch.device] = None,
) -> HSTURuntimeDevices:
    """Resolve the single runtime policy for dense and embedding phases.

    Current conservative policy:
    - dense phase follows the active accelerator or an explicit override
    - embedding phase defaults to CPU unless explicitly overridden

    The embedding decision intentionally does not infer from model weights.
    It must come from an explicit runtime choice or the conservative default.
    """
    if explicit_dense_device is not None:
        dense_device = torch.device(explicit_dense_device)
    else:
        dense_device = _infer_hstu_dense_device(module)

    if explicit_embedding_device is not None:
        embedding_device = torch.device(explicit_embedding_device)
    else:
        embedding_device = torch.device("cpu")

    return HSTURuntimeDevices(
        dense_device=dense_device,
        embedding_device=embedding_device,
    )


def resolve_hstu_sparse_transfer_plan(
    *,
    source_device: torch.device,
    target_device: torch.device,
) -> HSTUSparseTransferPlan:
    """Resolve whether sparse outputs need an explicit device transfer."""
    normalized_source = torch.device(source_device)
    normalized_target = torch.device(target_device)
    return HSTUSparseTransferPlan(
        source_device=normalized_source,
        target_device=normalized_target,
        requires_transfer=normalized_source != normalized_target,
    )


def _infer_hstu_dense_device(
    module: Optional[torch.nn.Module],
) -> torch.device:
    if module is None:
        return torch.device("cpu")

    for name, param in module.named_parameters():
        if not _is_embedding_tensor(name):
            return param.device

    for name, buffer in module.named_buffers():
        if not _is_embedding_tensor(name):
            return buffer.device

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_hstu_dense_device(
    module: torch.nn.Module,
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    """Resolve the accelerator device used by the dense HSTU path."""
    return resolve_hstu_runtime_devices(
        module,
        explicit_dense_device=explicit_device,
    ).dense_device


def resolve_hstu_embedding_device(
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    """Resolve the embedding device for the conservative phase-1 runtime."""
    return resolve_hstu_runtime_devices(
        explicit_embedding_device=explicit_device,
    ).embedding_device


def move_hstu_sparse_outputs_to_device(
    seq_embeddings: Dict[str, Any],
    payload_features: Dict[str, torch.Tensor],
    uih_seq_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    device: torch.device,
    *,
    source_device: Optional[torch.device] = None,
) -> Tuple[
    Dict[str, Any],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Move sparse preprocess outputs onto the dense execution device.

    The transfer contract is explicit:
    - if phase-1 and dense phase use the same device, this is a no-op
    - otherwise tensors are copied from source_device to the dense target
    - non-CPU dense targets may cast embeddings to bf16 for the dense path
    """
    from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import SequenceEmbedding

    resolved_source = torch.device(source_device or device)
    resolved_target = torch.device(device)
    transfer_plan = resolve_hstu_sparse_transfer_plan(
        source_device=resolved_source,
        target_device=resolved_target,
    )

    if not transfer_plan.requires_transfer:
        return (
            seq_embeddings,
            payload_features,
            uih_seq_lengths,
            num_candidates,
        )

    embedding_dtype = (
        torch.bfloat16 if transfer_plan.target_device.type != "cpu" else None
    )

    moved_seq_embeddings = {
        key: SequenceEmbedding(
            lengths=value.lengths.to(transfer_plan.target_device),
            embedding=value.embedding.to(
                device=transfer_plan.target_device,
                dtype=embedding_dtype or value.embedding.dtype,
            ),
        )
        for key, value in seq_embeddings.items()
    }
    moved_payload_features = {
        key: value.to(transfer_plan.target_device)
        for key, value in payload_features.items()
    }

    return (
        moved_seq_embeddings,
        moved_payload_features,
        uih_seq_lengths.to(transfer_plan.target_device),
        num_candidates.to(transfer_plan.target_device),
    )


def get_hstu_autocast_context(
    tensor: torch.Tensor,
    *,
    enabled: bool,
    dtype: torch.dtype = torch.bfloat16,
):
    """Return a conservative autocast context for the tensor's device."""
    if not enabled or tensor.device.type == "cpu":
        return nullcontext()

    try:
        return torch.autocast(
            device_type=tensor.device.type,
            dtype=dtype,
            enabled=True,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return nullcontext()
