# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Platform plugin ordering regression tests for vllm-GR."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
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


@contextmanager
def _fake_vllm(device_type: str):
    saved = _purge_modules("vllm")

    fake_vllm = types.ModuleType("vllm")
    fake_engine_pkg = types.ModuleType("vllm.engine")
    fake_engine_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    fake_platforms = types.ModuleType("vllm.platforms")

    @dataclass
    class FakeEngineArgs:
        model: str = ""
        trust_remote_code: bool = False
        max_model_len: int | None = None
        skip_tokenizer_init: bool = False
        runner: str = ""
        enforce_eager: bool = False
        enable_prefix_caching: bool = True
        worker_cls: str = ""
        dtype: str = "auto"
        tensor_parallel_size: int = 1
        hf_overrides: dict | None = None
        load_format: str | None = None

        def __post_init__(self):
            return None

        def create_engine_config(self, usage_context=None):
            return {"usage_context": usage_context}

    fake_engine_arg_utils.EngineArgs = FakeEngineArgs
    fake_platforms.current_platform = types.SimpleNamespace(device_type=device_type)

    fake_vllm.engine = fake_engine_pkg
    fake_vllm.platforms = fake_platforms
    fake_engine_pkg.arg_utils = fake_engine_arg_utils

    sys.modules.update({
        "vllm": fake_vllm,
        "vllm.engine": fake_engine_pkg,
        "vllm.engine.arg_utils": fake_engine_arg_utils,
        "vllm.platforms": fake_platforms,
    })

    try:
        yield fake_platforms.current_platform
    finally:
        _purge_modules("vllm")
        _restore_modules(saved)


@contextmanager
def _fake_vllm_ascend():
    saved = _purge_modules("vllm_ascend")

    fake_pkg = types.ModuleType("vllm_ascend")
    fake_platform = types.ModuleType("vllm_ascend.platform")
    state = {"register_calls": 0}

    class NPUPlatform:
        device_type = "npu"

    def register():
        state["register_calls"] += 1
        return "vllm_ascend.platform.NPUPlatform"

    fake_pkg.register = register
    fake_pkg._state = state
    fake_platform.NPUPlatform = NPUPlatform

    sys.modules.update({
        "vllm_ascend": fake_pkg,
        "vllm_ascend.platform": fake_platform,
    })

    try:
        yield state
    finally:
        _purge_modules("vllm_ascend")
        _restore_modules(saved)


def _load_arg_utils_module(test_name: str):
    module_path = ROOT / "vllm_gr" / "engine" / "arg_utils.py"
    module_name = f"_test_vllm_gr_arg_utils_plugin_order_{test_name}"
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


def _activate_registered_platform(current_platform: types.SimpleNamespace) -> str:
    target = import_module("vllm_ascend").register()
    module_name, class_name = target.rsplit(".", 1)
    platform_cls = getattr(import_module(module_name), class_name)
    current_platform.device_type = platform_cls.device_type
    return target


class TestPluginOrder(unittest.TestCase):

    def test_gpu_platform_without_npu_plugin_keeps_gpu_worker(self):
        prefixes = ("vllm", "vllm_gr", "vllm_ascend")
        with _clean_state(*prefixes), _fake_vllm("cuda"), _fake_vllm_ascend():
            module = _load_arg_utils_module("gpu_default")
            args = module.GREngineArgs(model="test-model")

            self.assertEqual(
                args.worker_cls,
                "vllm_gr.worker.gr_worker.GRWorker",
            )

    def test_platform_plugin_loaded_first_selects_npu_worker(self):
        prefixes = ("vllm", "vllm_gr", "vllm_ascend")
        with _clean_state(*prefixes), _fake_vllm("cuda") as current_platform, _fake_vllm_ascend() as state:
            target = _activate_registered_platform(current_platform)
            module = _load_arg_utils_module("npu_after_plugin")
            args = module.GREngineArgs(model="test-model")

            self.assertEqual(target, "vllm_ascend.platform.NPUPlatform")
            self.assertEqual(state["register_calls"], 1)
            self.assertEqual(current_platform.device_type, "npu")
            self.assertEqual(
                args.worker_cls,
                "vllm_gr.worker.gr_npu_worker.GRNPUWorker",
            )

    def test_worker_selection_tracks_platform_state_at_instantiation_time(self):
        prefixes = ("vllm", "vllm_gr", "vllm_ascend")
        with _clean_state(*prefixes), _fake_vllm("cuda") as current_platform, _fake_vllm_ascend():
            module = _load_arg_utils_module("order_sensitive")

            before_plugin = module.GREngineArgs(model="test-model")
            _activate_registered_platform(current_platform)
            after_plugin = module.GREngineArgs(model="test-model")

            self.assertEqual(
                before_plugin.worker_cls,
                "vllm_gr.worker.gr_worker.GRWorker",
            )
            self.assertEqual(
                after_plugin.worker_cls,
                "vllm_gr.worker.gr_npu_worker.GRNPUWorker",
            )

    def test_repeated_platform_plugin_activation_is_stable(self):
        prefixes = ("vllm", "vllm_gr", "vllm_ascend")
        with _clean_state(*prefixes), _fake_vllm("cuda") as current_platform, _fake_vllm_ascend() as state:
            module = _load_arg_utils_module("repeatable")

            first_target = _activate_registered_platform(current_platform)
            second_target = _activate_registered_platform(current_platform)
            args = module.GREngineArgs(model="test-model")

            self.assertEqual(first_target, second_target)
            self.assertEqual(state["register_calls"], 2)
            self.assertEqual(current_platform.device_type, "npu")
            self.assertEqual(
                args.worker_cls,
                "vllm_gr.worker.gr_npu_worker.GRNPUWorker",
            )


if __name__ == "__main__":
    unittest.main()
