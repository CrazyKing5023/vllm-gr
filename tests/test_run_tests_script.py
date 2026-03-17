# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Regression tests for scripts/run_tests.py suite selection."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_tests.py"


def _load_run_tests_module():
    module_name = "_test_vllm_gr_run_tests"
    spec = util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


class TestRunTestsScript(unittest.TestCase):

    def test_platform_gpu_suite_expands_to_plugin_shutdown_and_gpu_smoke(self):
        module = _load_run_tests_module()
        self.assertEqual(
            module.resolve_suites(["platform-gpu"]),
            ["plugin", "shutdown", "gpu-smoke"],
        )

    def test_platform_all_deduplicates_shared_suites(self):
        module = _load_run_tests_module()
        self.assertEqual(
            module.resolve_suites(["platform-all"]),
            [
                "plugin",
                "shutdown",
                "gpu-smoke",
                "npu-smoke",
                "cross-platform",
                "guardrails",
            ],
        )

    def test_examples_only_rejects_suite_selection(self):
        module = _load_run_tests_module()
        args = SimpleNamespace(examples_only=True, suite=["unit"], all=False)
        with self.assertRaises(ValueError):
            module.determine_requested_suites(args)

    def test_all_mode_includes_unit_and_platform_lanes(self):
        module = _load_run_tests_module()
        args = SimpleNamespace(examples_only=False, suite=None, all=True)
        self.assertEqual(
            module.determine_requested_suites(args),
            [
                "unit",
                "plugin",
                "shutdown",
                "gpu-smoke",
                "npu-smoke",
                "cross-platform",
                "guardrails",
            ],
        )


if __name__ == "__main__":
    unittest.main()
