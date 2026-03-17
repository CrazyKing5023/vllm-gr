#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Shared helpers for cross-platform example scripts."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Iterable, Optional


def add_repo_root_to_path(script_file: str) -> str:
    """Ensure the repository root is importable for local example execution."""
    script_dir = os.path.dirname(os.path.abspath(script_file))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: Optional[str] = None,
    include_model: bool = True,
) -> argparse.ArgumentParser:
    """Add shared runtime arguments used by example scripts."""
    if include_model:
        parser.add_argument(
            "--model",
            default=default_model,
            help="Model directory to load. Defaults to the example's built-in model path.",
        )
    parser.add_argument(
        "--expect-device",
        choices=["gpu", "cuda", "npu", "cpu"],
        default=None,
        help=(
            "Optional assertion for the active runtime backend. "
            "Use this to verify the example really ran on GPU or NPU."
        ),
    )
    return parser


def normalize_expected_device(device: Optional[str]) -> Optional[str]:
    """Map user-facing device labels to vLLM platform labels."""
    if device is None:
        return None
    normalized = device.lower()
    if normalized == "gpu":
        return "cuda"
    return normalized


def _first_non_empty_attr(obj: Any, attr_names: Iterable[str]) -> Any:
    """Return the first non-None attribute value from an object."""
    for attr_name in attr_names:
        if obj is None:
            return None
        value = getattr(obj, attr_name, None)
        if value is not None:
            return value
    return None


def _resolve_worker_and_runner(engine: Any) -> tuple[str, str]:
    """Best-effort runtime introspection for worker and model runner names."""
    worker_candidates: list[Any] = []

    for root_name in ("executor", "model_executor", "engine_core", "engine_core_client"):
        root = getattr(engine, root_name, None)
        if root is None:
            continue

        direct = _first_non_empty_attr(root, ("worker", "driver_worker"))
        if direct is not None:
            worker_candidates.append(direct)

        workers = getattr(root, "workers", None)
        if isinstance(workers, (list, tuple)) and workers:
            worker_candidates.extend(workers)

        worker_group = getattr(root, "worker_group", None)
        if worker_group is not None:
            nested = _first_non_empty_attr(worker_group, ("worker", "driver_worker"))
            if nested is not None:
                worker_candidates.append(nested)
            nested_workers = getattr(worker_group, "workers", None)
            if isinstance(nested_workers, (list, tuple)) and nested_workers:
                worker_candidates.extend(nested_workers)

    for worker in worker_candidates:
        if worker is None:
            continue
        worker_cls = type(worker).__name__
        model_runner = getattr(worker, "model_runner", None)
        if model_runner is not None:
            return worker_cls, type(model_runner).__name__
        return worker_cls, "unknown"

    configured_worker = None
    vllm_config = getattr(engine, "vllm_config", None)
    if vllm_config is not None:
        configured_worker = getattr(vllm_config, "worker_cls", None)
        if configured_worker is None:
            parallel_config = getattr(vllm_config, "parallel_config", None)
            configured_worker = getattr(parallel_config, "worker_cls", None)

    if isinstance(configured_worker, str) and configured_worker:
        worker_cls = configured_worker.rsplit(".", 1)[-1]
        runner_map = {
            "GRWorker": "GRModelRunner",
            "GRNPUWorker": "GRNPUModelRunner",
        }
        return f"{worker_cls} (configured)", runner_map.get(worker_cls, "unknown")

    return "unknown", "unknown"


def describe_runtime(llm, expected_device: Optional[str] = None) -> str:
    """Print the active vLLM runtime backend and verify it when requested."""
    from vllm.platforms import current_platform

    actual_device = str(getattr(current_platform, "device_type", "unknown"))

    engine = llm.get_engine()
    worker_cls, runner_cls = _resolve_worker_and_runner(engine)
    executor = getattr(engine, "executor", None)
    executor_cls = type(executor).__name__ if executor is not None else "unknown"

    configured_worker = "unknown"
    vllm_config = getattr(engine, "vllm_config", None)
    if vllm_config is not None:
        configured_worker = getattr(vllm_config, "worker_cls", None) or "unknown"
        if configured_worker == "unknown":
            parallel_config = getattr(vllm_config, "parallel_config", None)
            configured_worker = getattr(parallel_config, "worker_cls", None) or "unknown"

    print("\n[Runtime]")
    print(f"   Active platform: {actual_device}")
    print(f"   Executor: {executor_cls}")
    print(f"   GREngineArgs.worker_cls: {configured_worker}")
    print(f"   Worker: {worker_cls}")
    print(f"   Model runner: {runner_cls}")

    normalized_expected = normalize_expected_device(expected_device)
    if normalized_expected is not None and actual_device != normalized_expected:
        raise RuntimeError(
            "Example runtime backend mismatch: "
            f"expected {normalized_expected}, got {actual_device}. "
            "Check the current environment, installed platform plugins, and base runtime."
        )

    return actual_device
