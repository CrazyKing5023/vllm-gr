# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Import boundary regression tests for vllm-GR."""

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
def _clean_import_state(*prefixes: str):
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
        yield
    finally:
        _purge_modules("vllm")
        _restore_modules(saved)


def _load_arg_utils_module(device_type: str):
    module_path = ROOT / "vllm_gr" / "engine" / "arg_utils.py"
    module_name = f"_test_vllm_gr_arg_utils_{device_type}"
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestImportContract(unittest.TestCase):

    def test_top_level_import_does_not_load_vllm_ascend(self):
        with _clean_import_state("vllm_gr", "vllm_ascend"):
            module = import_module("vllm_gr")
            self.assertEqual(module.__version__, "0.3.0")
            self.assertNotIn("vllm_ascend", sys.modules)
            self.assertNotIn("vllm_gr.plugin", sys.modules)
            self.assertNotIn("vllm_gr.patch", sys.modules)

    def test_plugin_import_does_not_load_vllm_ascend(self):
        with _clean_import_state("vllm_gr", "vllm_ascend"):
            plugin = import_module("vllm_gr.plugin")
            self.assertTrue(hasattr(plugin, "initialize_runtime"))
            self.assertNotIn("vllm_ascend", sys.modules)

    def test_gpu_worker_selection_is_string_only(self):
        with _clean_import_state("vllm_gr", "vllm_ascend"), _fake_vllm("cuda"):
            module = _load_arg_utils_module("cuda")
            args = module.GREngineArgs(model="test-model")
            self.assertEqual(
                args.worker_cls,
                "vllm_gr.worker.gr_worker.GRWorker",
            )
            self.assertNotIn("vllm_ascend", sys.modules)
            self.assertNotIn("vllm_gr.worker.gr_npu_worker", sys.modules)

    def test_npu_worker_selection_does_not_import_vendor_worker_eagerly(self):
        with _clean_import_state("vllm_gr", "vllm_ascend"), _fake_vllm("npu"):
            module = _load_arg_utils_module("npu")
            args = module.GREngineArgs(model="test-model")
            self.assertEqual(
                args.worker_cls,
                "vllm_gr.worker.gr_npu_worker.GRNPUWorker",
            )
            self.assertNotIn("vllm_ascend", sys.modules)
            self.assertNotIn("vllm_gr.worker.gr_npu_worker", sys.modules)


if __name__ == "__main__":
    unittest.main()
