# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Shutdown-chain regression tests for GRLLMEngine."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import util
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
def _fake_runtime():
    saved = _purge_modules("vllm", "vllm_gr")
    trace: list[str] = []
    destroy_calls: list[object] = []

    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_usage_pkg = types.ModuleType("vllm.usage")
    fake_usage_lib = types.ModuleType("vllm.usage.usage_lib")
    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_distributed = types.ModuleType("vllm.distributed")
    fake_v1 = types.ModuleType("vllm.v1")
    fake_v1_engine_pkg = types.ModuleType("vllm.v1.engine")
    fake_v1_engine_llm = types.ModuleType("vllm.v1.engine.llm_engine")
    fake_v1_executor = types.ModuleType("vllm.v1.executor")
    fake_v1_metrics = types.ModuleType("vllm.v1.metrics")
    fake_v1_metrics_loggers = types.ModuleType("vllm.v1.metrics.loggers")

    fake_gr_config = types.ModuleType("vllm_gr.config.gr_config")
    fake_arg_utils = types.ModuleType("vllm_gr.engine.arg_utils")
    fake_input_processor = types.ModuleType("vllm_gr.engine.gr_input_processor")
    fake_output_processor = types.ModuleType("vllm_gr.engine.gr_output_processor")

    class FakeLLMEngine:
        def shutdown(self):
            trace.append("base.shutdown")

    class FakeExecutor:
        pass

    class UsageContext:
        ENGINE_CONTEXT = "engine"

    class MultiModalRegistry:
        pass

    class StatLoggerFactory:
        pass

    class GRConfig:
        def __init__(self, max_seq_len=512, max_candidates=1000, default_top_k=10):
            self.max_seq_len = max_seq_len
            self.max_candidates = max_candidates
            self.default_top_k = default_top_k

    class GREngineArgs:
        pass

    class GRInputProcessor:
        pass

    class GROutputProcessor:
        pass

    def destroy_group(group):
        destroy_calls.append(group)
        trace.append("dp_group.destroy")

    fake_config.VllmConfig = object
    fake_usage_lib.UsageContext = UsageContext
    fake_multimodal.MULTIMODAL_REGISTRY = object()
    fake_multimodal.MultiModalRegistry = MultiModalRegistry
    fake_distributed.stateless_destroy_torch_distributed_process_group = destroy_group
    fake_v1_engine_llm.LLMEngine = FakeLLMEngine
    fake_v1_executor.Executor = FakeExecutor
    fake_v1_metrics_loggers.StatLoggerFactory = StatLoggerFactory
    fake_gr_config.GRConfig = GRConfig
    fake_arg_utils.GREngineArgs = GREngineArgs
    fake_input_processor.GRInputProcessor = GRInputProcessor
    fake_output_processor.GROutputProcessor = GROutputProcessor

    sys.modules.update({
        "vllm": fake_vllm,
        "vllm.config": fake_config,
        "vllm.usage": fake_usage_pkg,
        "vllm.usage.usage_lib": fake_usage_lib,
        "vllm.multimodal": fake_multimodal,
        "vllm.distributed": fake_distributed,
        "vllm.v1": fake_v1,
        "vllm.v1.engine": fake_v1_engine_pkg,
        "vllm.v1.engine.llm_engine": fake_v1_engine_llm,
        "vllm.v1.executor": fake_v1_executor,
        "vllm.v1.metrics": fake_v1_metrics,
        "vllm.v1.metrics.loggers": fake_v1_metrics_loggers,
        "vllm_gr.config.gr_config": fake_gr_config,
        "vllm_gr.engine.arg_utils": fake_arg_utils,
        "vllm_gr.engine.gr_input_processor": fake_input_processor,
        "vllm_gr.engine.gr_output_processor": fake_output_processor,
    })

    try:
        yield trace, destroy_calls
    finally:
        _purge_modules("vllm", "vllm_gr")
        _restore_modules(saved)


def _load_gr_engine_module():
    module_path = ROOT / "vllm_gr" / "engine" / "gr_engine.py"
    module_name = "_test_vllm_gr_engine_shutdown_chain"
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


class _Component:
    def __init__(
        self,
        name: str,
        trace: list[str],
        *,
        method_name: str = "shutdown",
        fail: bool = False,
        **children,
    ):
        self.name = name
        self._trace = trace
        self._method_name = method_name
        self._fail = fail
        for attr_name, child in children.items():
            setattr(self, attr_name, child)

    def shutdown(self):
        self._trace.append(f"{self.name}.shutdown")
        if self._fail:
            raise RuntimeError(f"{self.name} boom")

    def close(self):
        self._trace.append(f"{self.name}.close")
        if self._fail:
            raise RuntimeError(f"{self.name} boom")


class TestShutdownChain(unittest.TestCase):

    def test_shutdown_releases_engine_chain_and_process_group(self):
        prefixes = ("vllm", "vllm_gr")
        with _clean_state(*prefixes), _fake_runtime() as (trace, destroy_calls):
            module = _load_gr_engine_module()

            scheduler = _Component("scheduler", trace)
            driver_worker = _Component("driver_worker", trace)
            worker_group = _Component("worker_group", trace)
            model_executor = _Component("model_executor", trace)
            executor = _Component(
                "executor",
                trace,
                driver_worker=driver_worker,
                workers=[worker_group],
            )
            engine_core = _Component(
                "engine_core",
                trace,
                model_executor=model_executor,
                executor=executor,
                scheduler=scheduler,
            )
            logger_manager = _Component("logger_manager", trace, method_name="close")
            dp_group = object()

            engine = module.GRLLMEngine.__new__(module.GRLLMEngine)
            engine._shutdown_complete = False
            engine._num_gr_requests_processed = 2
            engine.engine_core = engine_core
            engine.engine_core_client = None
            engine.model_executor = model_executor
            engine.executor = executor
            engine.logger_manager = logger_manager
            engine.dp_group = dp_group
            engine.input_processor = object()
            engine.output_processor = object()
            engine.io_processor = object()
            engine.tokenizer = object()

            engine.shutdown()

            self.assertEqual(
                trace,
                [
                    "base.shutdown",
                    "engine_core.shutdown",
                    "model_executor.shutdown",
                    "executor.shutdown",
                    "driver_worker.shutdown",
                    "worker_group.shutdown",
                    "scheduler.shutdown",
                    "logger_manager.shutdown",
                    "dp_group.destroy",
                ],
            )
            self.assertEqual(destroy_calls, [dp_group])
            self.assertIsNone(engine.engine_core)
            self.assertIsNone(engine.model_executor)
            self.assertIsNone(engine.executor)
            self.assertIsNone(engine.logger_manager)
            self.assertIsNone(engine.dp_group)
            self.assertIsNone(engine.input_processor)
            self.assertIsNone(engine.output_processor)
            self.assertIsNone(engine.io_processor)
            self.assertIsNone(engine.tokenizer)

    def test_shutdown_continues_after_component_failure_and_is_idempotent(self):
        prefixes = ("vllm", "vllm_gr")
        with _clean_state(*prefixes), _fake_runtime() as (trace, destroy_calls):
            module = _load_gr_engine_module()

            model_executor = _Component("model_executor", trace)
            executor = _Component("executor", trace)
            engine_core = _Component(
                "engine_core",
                trace,
                fail=True,
                model_executor=model_executor,
                executor=executor,
            )
            logger_manager = _Component("logger_manager", trace)
            dp_group = object()

            engine = module.GRLLMEngine.__new__(module.GRLLMEngine)
            engine._shutdown_complete = False
            engine._num_gr_requests_processed = 1
            engine.engine_core = engine_core
            engine.engine_core_client = None
            engine.model_executor = model_executor
            engine.executor = executor
            engine.logger_manager = logger_manager
            engine.dp_group = dp_group
            engine.input_processor = object()
            engine.output_processor = object()
            engine.io_processor = object()
            engine.tokenizer = object()

            engine.shutdown()
            first_trace = list(trace)
            engine.shutdown()

            self.assertEqual(
                first_trace,
                [
                    "base.shutdown",
                    "engine_core.shutdown",
                    "model_executor.shutdown",
                    "executor.shutdown",
                    "logger_manager.shutdown",
                    "dp_group.destroy",
                ],
            )
            self.assertEqual(trace, first_trace)
            self.assertEqual(destroy_calls, [dp_group])


if __name__ == "__main__":
    unittest.main()
