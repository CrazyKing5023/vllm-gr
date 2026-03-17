#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Unified test runner for vllm-GR.

Runs core pytest suites, platform-specific smoke lanes, and example scripts.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --suite platform-gpu
    python scripts/run_tests.py --suite platform-npu cross-platform
    python scripts/run_tests.py --list-suites
    python scripts/run_tests.py --all
    python scripts/run_tests.py --examples-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")


SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_engine.py",
            "test_processors.py",
            "test_weight_loading.py",
            "test_npu_adaptation.py",
            "test_hstu_runtime_policy.py",
        ],
    },
    "plugin": {
        "label": "Plugin and Import Contract Tests",
        "runner": "python",
        "files": [
            "test_import_contract.py",
            "test_patch_chain.py",
            "test_gr_patch_chain.py",
            "test_plugin_order.py",
        ],
    },
    "shutdown": {
        "label": "Shutdown Chain Tests",
        "runner": "python",
        "files": [
            "test_shutdown_chain.py",
        ],
    },
    "gpu-smoke": {
        "label": "GPU Smoke Tests",
        "runner": "python",
        "files": [
            "test_gpu_smoke.py",
        ],
    },
    "npu-smoke": {
        "label": "NPU Smoke Tests",
        "runner": "python",
        "files": [
            "test_npu_smoke.py",
        ],
    },
    "cross-platform": {
        "label": "Cross-Platform Cleanup Tests",
        "runner": "python",
        "files": [
            "test_cross_platform_cleanup.py",
        ],
    },
    "guardrails": {
        "label": "Platform Guardrail Tests",
        "runner": "python",
        "files": [
            "test_platform_guardrails.py",
        ],
    },
    "platform-gpu": {
        "label": "GPU Platform Lane",
        "includes": ["plugin", "shutdown", "gpu-smoke"],
    },
    "platform-npu": {
        "label": "NPU Platform Lane",
        "includes": ["plugin", "shutdown", "npu-smoke"],
    },
    "platform-all": {
        "label": "All Platform Lanes",
        "includes": ["platform-gpu", "platform-npu", "cross-platform", "guardrails"],
    },
}


def run_command(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label}  ({elapsed:.1f}s)")
    return result.returncode == 0


def _expand_suite(
    suite_name: str,
    expanded: list[str],
    seen: set[str],
    stack: set[str],
) -> None:
    if suite_name in seen:
        return
    if suite_name in stack:
        raise ValueError(f"Cyclic suite definition detected: {suite_name}")
    if suite_name not in SUITE_DEFINITIONS:
        raise KeyError(f"Unknown suite: {suite_name}")

    stack.add(suite_name)
    suite = SUITE_DEFINITIONS[suite_name]
    for included in suite.get("includes", []):
        _expand_suite(str(included), expanded, seen, stack)
    stack.remove(suite_name)

    if "files" in suite:
        expanded.append(suite_name)
        seen.add(suite_name)


def resolve_suites(selected: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for suite_name in selected:
        _expand_suite(suite_name, expanded, seen, set())
    return expanded


def build_pytest_command(files: list[str], verbose: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(os.path.join(TESTS_DIR, file_name) for file_name in files)
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])
    return cmd


def run_pytest_suite(suite_name: str, verbose: bool = False) -> bool:
    suite = SUITE_DEFINITIONS[suite_name]
    files = suite.get("files", [])
    if not files:
        raise ValueError(f"Suite {suite_name!r} does not define files")
    cmd = build_pytest_command(list(files), verbose=verbose)
    return run_command(cmd, str(suite["label"]))


def run_python_suite(suite_name: str) -> bool:
    suite = SUITE_DEFINITIONS[suite_name]
    files = suite.get("files", [])
    if not files:
        raise ValueError(f"Suite {suite_name!r} does not define files")

    results = []
    for file_name in files:
        cmd = [sys.executable, os.path.join(TESTS_DIR, file_name)]
        results.append(run_command(cmd, f"{suite['label']}: {file_name}"))
    return all(results)


def run_example(script: str, extra_args: list[str] | None = None) -> bool:
    path = os.path.join(EXAMPLES_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {script} not found")
        return True

    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    return run_command(cmd, f"Example: {script}")


def run_examples(datasets: list[str] | None = None) -> bool:
    results = []

    results.append(run_example("simple_inference.py"))

    if datasets:
        for ds in datasets:
            results.append(run_example("dataset_inference.py", [ds]))
    else:
        results.append(run_example("dataset_inference.py", ["debug"]))

    return all(results)


def print_suite_catalog() -> None:
    print("=" * 60)
    print("  Available Test Suites")
    print("=" * 60)
    for name, suite in SUITE_DEFINITIONS.items():
        label = suite["label"]
        runner = suite.get("runner", "composite")
        if "files" in suite:
            files = ", ".join(suite["files"])
            print(f"  {name:15s} {label}")
            print(f"  runner: {runner}")
            print(f"  files: {files}")
        else:
            includes = ", ".join(suite.get("includes", []))
            print(f"  {name:15s} {label}")
            print(f"  includes: {includes}")
    print("=" * 60)


def determine_requested_suites(args: argparse.Namespace) -> list[str]:
    if args.examples_only:
        if args.suite:
            raise ValueError("--examples-only cannot be combined with --suite")
        return []

    if args.suite:
        return resolve_suites(args.suite)

    if args.all:
        return resolve_suites(["unit", "platform-all"])

    return resolve_suites(["unit"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vllm-GR tests and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run core unit tests, all platform lanes, and examples",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Run example scripts only (skip pytest suites)",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=sorted(SUITE_DEFINITIONS),
        help=(
            "Run one or more named pytest suites. "
            "Examples: unit, plugin, gpu-smoke, npu-smoke, "
            "platform-gpu, platform-npu, platform-all, cross-platform"
        ),
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print all available pytest suite names and exit",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help=(
            "Dataset(s) for dataset_inference.py "
            "(e.g. debug, movielens-1m, kuairand-1k, streaming)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_suites:
        print_suite_catalog()
        return

    requested_suites = determine_requested_suites(args)

    print("=" * 60)
    print("  vllm-GR Test Runner")
    print("=" * 60)

    if requested_suites:
        print(f"  Selected suites: {', '.join(requested_suites)}")
    elif args.examples_only:
        print("  Selected suites: examples-only")

    results: list[tuple[str, bool]] = []
    start = time.time()

    for suite_name in requested_suites:
        suite_label = str(SUITE_DEFINITIONS[suite_name]["label"])
        runner = SUITE_DEFINITIONS[suite_name].get("runner", "pytest")
        if runner == "python":
            passed = run_python_suite(suite_name)
        else:
            passed = run_pytest_suite(suite_name, args.verbose)
        results.append((suite_label, passed))

    if args.all or args.examples_only:
        results.append(("Examples", run_examples(args.dataset)))

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  Summary  ({elapsed:.1f}s total)")
    print(f"{'=' * 60}")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {status:8s}  {name}")
        if not passed:
            all_passed = False

    print(f"{'=' * 60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
