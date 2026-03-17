# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Regression tests for static platform guardrails."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_platform_guardrails.py"


def _load_guardrail_module():
    module_name = "_test_vllm_gr_platform_guardrails"
    spec = util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


class TestPlatformGuardrails(unittest.TestCase):

    def test_current_repo_passes_platform_guardrails(self):
        module = _load_guardrail_module()
        violations = module.collect_guardrail_violations(ROOT)
        self.assertEqual(violations, [])

    def test_guardrails_flag_shared_layer_vendor_imports_and_probes(self):
        module = _load_guardrail_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            shared_file = repo_root / "vllm_gr" / "models" / "bad_shared.py"
            shared_file.parent.mkdir(parents=True, exist_ok=True)
            shared_file.write_text(
                textwrap.dedent(
                    """
                    import torch
                    import torch_npu

                    def pick_device():
                        if torch.cuda.is_available():
                            return "cuda"
                        return torch.ops.mxrec.hstu_dense
                    """
                ).strip()
                + "\n"
            )

            violations = module.collect_guardrail_violations(repo_root)
            details = {(violation.rule, violation.detail) for violation in violations}

            self.assertIn(("torch_npu", "direct import: torch_npu"), details)
            self.assertIn(("is_available", "torch.cuda.is_available()"), details)
            self.assertIn(("mxrec", "torch.ops.mxrec.hstu_dense"), details)

    def test_arg_utils_current_platform_usage_is_allowlisted(self):
        module = _load_guardrail_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            arg_utils = repo_root / "vllm_gr" / "engine" / "arg_utils.py"
            arg_utils.parent.mkdir(parents=True, exist_ok=True)
            arg_utils.write_text(
                textwrap.dedent(
                    """
                    from vllm.platforms import current_platform

                    def is_npu():
                        return current_platform.device_type == "npu"
                    """
                ).strip()
                + "\n"
            )

            violations = module.collect_guardrail_violations(repo_root)
            self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
