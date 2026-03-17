# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""General plugin coexistence tests for vllm-GR and vllm_ascend."""

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
    fake_config_pkg = types.ModuleType("vllm_gr.config")
    fake_hstu_cfg = types.ModuleType("vllm_gr.config.hstu_hf_config")
    fake_model_executor = types.ModuleType("vllm_gr.model_executor")
    fake_models_pkg = types.ModuleType("vllm_gr.model_executor.models")
    fake_registry = types.ModuleType("vllm_gr.model_executor.models.registry")

    class GREngineCoreRequest(originals["EngineCoreRequest"]):
        pass

    class GRRequest(originals["Request"]):
        pass

    class GRNewRequestData(originals["NewRequestData"]):
        pass

    fake_engine.GREngineCoreRequest = GREngineCoreRequest
    fake_request.GRRequest = GRRequest
    fake_sched_output.GRNewRequestData = GRNewRequestData
    fake_hstu_cfg.ensure_hstu_config_registered = lambda: True
    fake_registry.register_gr_models_to_vllm = lambda: True

    sys.modules.update({
        "vllm_gr.engine": fake_engine,
        "vllm_gr.request": fake_request,
        "vllm_gr.core": fake_core_pkg,
        "vllm_gr.core.sched": fake_sched_pkg,
        "vllm_gr.core.sched.output": fake_sched_output,
        "vllm_gr.config": fake_config_pkg,
        "vllm_gr.config.hstu_hf_config": fake_hstu_cfg,
        "vllm_gr.model_executor": fake_model_executor,
        "vllm_gr.model_executor.models": fake_models_pkg,
        "vllm_gr.model_executor.models.registry": fake_registry,
    })

    return {
        "GREngineCoreRequest": GREngineCoreRequest,
        "GRRequest": GRRequest,
        "GRNewRequestData": GRNewRequestData,
    }


def _install_fake_vllm_ascend_modules():
    fake_pkg = types.ModuleType("vllm_ascend")

    def register_connector():
        # Simulate another general plugin loading critical consumers early.
        __import__("vllm.v1.engine.core")
        __import__("vllm.v1.core.sched.scheduler")

    def register_model_loader():
        late_consumer = types.ModuleType("vllm_ascend.fake_general_consumer")
        late_consumer.EngineCoreRequest = sys.modules["vllm.v1.engine"].EngineCoreRequest
        late_consumer.Request = sys.modules["vllm.v1.request"].Request
        late_consumer.NewRequestData = sys.modules["vllm.v1.core.sched.output"].NewRequestData
        sys.modules["vllm_ascend.fake_general_consumer"] = late_consumer

    def register_service_profiling():
        return None

    fake_pkg.register_connector = register_connector
    fake_pkg.register_model_loader = register_model_loader
    fake_pkg.register_service_profiling = register_service_profiling
    sys.modules["vllm_ascend"] = fake_pkg


class TestGRPatchChain(unittest.TestCase):

    def test_general_plugin_coexistence_keeps_gr_extended_types(self):
        prefixes = (
            "vllm",
            "vllm_gr.patch",
            "vllm_gr.plugin",
            "vllm_gr.engine",
            "vllm_gr.request",
            "vllm_gr.core",
            "vllm_gr.config",
            "vllm_gr.model_executor",
            "vllm_ascend",
        )
        with _clean_state(*prefixes):
            originals = _install_fake_vllm_modules()
            replacements = _install_fake_gr_modules(originals)
            _install_fake_vllm_ascend_modules()

            ascend = importlib.import_module("vllm_ascend")
            ascend.register_connector()

            plugin = importlib.import_module("vllm_gr.plugin")
            state = plugin.initialize_runtime()

            ascend.register_model_loader()
            from vllm_gr.patch import verify_patches

            self.assertGreaterEqual(state["patch_count"], 1)
            self.assertEqual(verify_patches(), [])
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

            late_consumer = sys.modules["vllm_ascend.fake_general_consumer"]
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
