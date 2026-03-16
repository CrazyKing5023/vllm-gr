#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Unified test runner for vllm-GR.

Runs pytest unit tests and (optionally) example scripts to validate the
full inference pipeline.

Usage:
    python scripts/run_tests.py                   # unit tests only
    python scripts/run_tests.py --all             # unit tests + examples
    python scripts/run_tests.py --examples-only   # examples only
    python scripts/run_tests.py --dataset debug   # single dataset example
"""

import argparse
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")


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


def run_unit_tests(verbose: bool = False) -> bool:
    """Run pytest unit tests under tests/."""
    cmd = [sys.executable, "-m", "pytest", TESTS_DIR]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])
    return run_command(cmd, "Unit Tests (pytest)")


def run_example(script: str, extra_args: list[str] | None = None) -> bool:
    """Run a single example script."""
    path = os.path.join(EXAMPLES_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {script} not found")
        return True

    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    return run_command(cmd, f"Example: {script}")


def run_examples(datasets: list[str] | None = None) -> bool:
    """Run example scripts."""
    results = []

    results.append(run_example("simple_inference.py"))

    if datasets:
        for ds in datasets:
            results.append(
                run_example("dataset_inference.py", [ds])
            )
    else:
        results.append(run_example("dataset_inference.py", ["debug"]))

    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run vllm-GR tests and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both unit tests and examples",
    )
    parser.add_argument(
        "--examples-only", action="store_true",
        help="Run example scripts only (skip pytest)",
    )
    parser.add_argument(
        "--dataset", nargs="*",
        help="Dataset(s) for dataset_inference.py "
             "(e.g. debug, movielens-1m, kuairand-1k, streaming)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  vllm-GR Test Runner")
    print("=" * 60)

    results = []
    start = time.time()

    if not args.examples_only:
        results.append(("Unit Tests", run_unit_tests(args.verbose)))

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
