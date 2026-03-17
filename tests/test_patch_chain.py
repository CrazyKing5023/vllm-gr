# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Patch chain tests for vllm-GR."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import importlib
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _purge_modules(*prefixes: str) -> dict[str, types.ModuleType]:
    saved: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            saved[name] = module
            sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


@contextmanager
def _clean_state(*prefixes: str):
    saved = _purge_modules(*prefixes)
    try:
        yield
    finally:
        _purge_modules(*prefixes)
        _restore_modules(saved)


def _install_fake_vllm_modules():
    fake_vllm = types.ModuleType("vllm")
    fake_v1 = types.ModuleType("vllm.v1")
    fake_engine = types.ModuleType("vllm.v1.engine")
    fake_engine_core = types.ModuleType("vllm.v1.engine.core")
    fake_request = types.ModuleType("vllm.v1.request")
    fake_core_pkg = types.ModuleType("vllm.v1.core")
    fake_sched_pkg = types.ModuleType("vllm.v1.core.sched")
    fake_sched_output = types.ModuleType("vllm.v1.core.sched.output")
    fake_sched_scheduler = types.ModuleType("vllm.v1.core.sched.scheduler")

    class EngineCoreRequest:
        pass

    class Request:
        pass

    class NewRequestData:
        pass

    fake_engine.EngineCoreRequest = EngineCoreRequest
    fake_engine_core.EngineCoreRequest = EngineCoreRequest
    fake_engine_core.Request = Request
    fake_request.Request = Request
    fake_sched_output.NewRequestData = NewRequestData
    fake_sched_scheduler.NewRequestData = NewRequestData

    sys.modules.update({
        "vllm": fake_vllm,
        "vllm.v1": fake_v1,
        "vllm.v1.engine": fake_engine,
        "vllm.v1.engine.core": fake_engine_core,
        "vllm.v1.request": fake_request,
        "vllm.v1.core": fake_core_pkg,
        "vllm.v1.core.sched": fake_sched_pkg,
        "vllm.v1.core.sched.output": fake_sched_output,
        "vllm.v1.core.sched.scheduler": fake_sched_scheduler,
    })

    return {
        "EngineCoreRequest": EngineCoreRequest,
        "Request": Request,
        "NewRequestData": NewRequestData,
        "engine_module": fake_engine,
        "engine_core_module": fake_engine_core,
        "request_module": fake_request,
        "sched_output_module": fake_sched_output,
        "sched_scheduler_module": fake_sched_scheduler,
    }


def _install_fake_gr_modules(originals: dict[str, object]):
    fake_engine = types.ModuleType("vllm_gr.engine")
    fake_request = types.ModuleType("vllm_gr.request")
    fake_core_pkg = types.ModuleType("vllm_gr.core")
    fake_sched_pkg = types.ModuleType("vllm_gr.core.sched")
    fake_sched_output = types.ModuleType("vllm_gr.core.sched.output")

    class GREngineCoreRequest(originals["EngineCoreRequest"]):
        pass

    class GRRequest(originals["Request"]):
        pass

    class GRNewRequestData(originals["NewRequestData"]):
        pass

    fake_engine.GREngineCoreRequest = GREngineCoreRequest
    fake_request.GRRequest = GRRequest
    fake_sched_output.GRNewRequestData = GRNewRequestData

    sys.modules.update({
        "vllm_gr.engine": fake_engine,
        "vllm_gr.request": fake_request,
        "vllm_gr.core": fake_core_pkg,
        "vllm_gr.core.sched": fake_sched_pkg,
        "vllm_gr.core.sched.output": fake_sched_output,
    })

    return {
        "GREngineCoreRequest": GREngineCoreRequest,
        "GRRequest": GRRequest,
        "GRNewRequestData": GRNewRequestData,
    }


class TestPatchChain(unittest.TestCase):

    def test_apply_patches_updates_sources_and_consumers(self):
        prefixes = ("vllm", "vllm_gr.patch", "vllm_gr.engine", "vllm_gr.request", "vllm_gr.core")
        with _clean_state(*prefixes):
            originals = _install_fake_vllm_modules()
            replacements = _install_fake_gr_modules(originals)

            patch = importlib.import_module("vllm_gr.patch")
            count = patch.apply_patches_if_needed()

            self.assertGreater(count, 0)
            self.assertIs(
                originals["engine_module"].EngineCoreRequest,
                replacements["GREngineCoreRequest"],
            )
            self.assertIs(
                originals["request_module"].Request,
                replacements["GRRequest"],
            )
            self.assertIs(
                originals["sched_output_module"].NewRequestData,
                replacements["GRNewRequestData"],
            )
            self.assertIs(
                originals["engine_core_module"].EngineCoreRequest,
                replacements["GREngineCoreRequest"],
            )
            self.assertIs(
                originals["engine_core_module"].Request,
                replacements["GRRequest"],
            )
            self.assertIs(
                originals["sched_scheduler_module"].NewRequestData,
                replacements["GRNewRequestData"],
            )
            self.assertEqual(patch.verify_patches(), [])

    def test_verify_patches_reports_broken_consumer(self):
        prefixes = ("vllm", "vllm_gr.patch", "vllm_gr.engine", "vllm_gr.request", "vllm_gr.core")
        with _clean_state(*prefixes):
            originals = _install_fake_vllm_modules()
            _install_fake_gr_modules(originals)

            patch = importlib.import_module("vllm_gr.patch")
            patch.apply_patches_if_needed()

            originals["engine_core_module"].Request = originals["Request"]
            with self.assertLogs("vllm_gr.patch", level="WARNING") as captured:
                warnings = patch.verify_patches()

            self.assertEqual(len(warnings), 1)
            self.assertIn("vllm.v1.engine.core.Request", warnings[0])
            self.assertEqual(len(captured.output), 1)

    def test_reapply_patches_catches_late_loaded_consumer(self):
        prefixes = ("vllm", "vllm_gr.patch", "vllm_gr.engine", "vllm_gr.request", "vllm_gr.core")
        with _clean_state(*prefixes):
            originals = _install_fake_vllm_modules()
            replacements = _install_fake_gr_modules(originals)

            patch = importlib.import_module("vllm_gr.patch")
            patch.apply_patches_if_needed()

            late_consumer = types.ModuleType("vllm.v1.worker.fake_consumer")
            late_consumer.EngineCoreRequest = originals["EngineCoreRequest"]
            late_consumer.Request = originals["Request"]
            late_consumer.NewRequestData = originals["NewRequestData"]
            sys.modules["vllm.v1.worker.fake_consumer"] = late_consumer

            patch.re_apply_patches()

            self.assertIs(
                late_consumer.EngineCoreRequest,
                replacements["GREngineCoreRequest"],
            )
            self.assertIs(late_consumer.Request, replacements["GRRequest"])
            self.assertIs(
                late_consumer.NewRequestData,
                replacements["GRNewRequestData"],
            )


if __name__ == "__main__":
    unittest.main()
