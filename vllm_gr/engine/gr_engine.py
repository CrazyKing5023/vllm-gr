# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLMEngine — extends vLLM's LLMEngine for recommendation inference.

After standard initialization, replaces the default input/output processors
with GR-specific versions (GRInputProcessor, GROutputProcessor) so the
existing add_request() / step() pipeline handles GRPrompt transparently.
"""

from typing import Any
import time
import logging

from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import Executor
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.metrics.loggers import StatLoggerFactory

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor

logger = logging.getLogger("vllm_gr.engine")


def _invoke_shutdown_method(component: Any) -> bool:
    """Best-effort shutdown for a runtime component."""
    for method_name in ("shutdown", "close", "terminate", "stop"):
        method = getattr(component, method_name, None)
        if callable(method):
            method()
            return True
    return False


def _iter_nested_components(component: Any):
    """Yield common nested runtime children for shutdown traversal."""
    for attr_name in (
        "engine_core",
        "engine_core_client",
        "model_executor",
        "executor",
        "scheduler",
        "structured_output_manager",
        "driver_worker",
        "worker",
        "workers",
    ):
        child = getattr(component, attr_name, None)
        if child is None:
            continue
        if isinstance(child, dict):
            iterable = child.values()
        elif isinstance(child, (list, tuple, set)):
            iterable = child
        else:
            iterable = (child,)
        for nested in iterable:
            yield attr_name, nested


def _shutdown_nested_component(
    label: str,
    component: Any,
    seen: set[int],
    errors: list[str],
) -> None:
    """Release a component and common nested runtime children once."""
    if component is None:
        return

    component_id = id(component)
    if component_id in seen:
        return
    seen.add(component_id)

    try:
        _invoke_shutdown_method(component)
    except Exception as exc:
        errors.append(f"{label}: {exc}")
        logger.warning("Shutdown step failed for %s", label, exc_info=True)

    for attr_name, nested in _iter_nested_components(component):
        _shutdown_nested_component(f"{label}.{attr_name}", nested, seen, errors)


def _destroy_dp_group(dp_group: Any, errors: list[str]) -> None:
    """Destroy the stateless DP process group when present."""
    if dp_group is None:
        return

    try:
        from vllm.distributed import (
            stateless_destroy_torch_distributed_process_group,
        )
    except Exception as exc:
        errors.append(f"dp_group_import: {exc}")
        logger.warning("Failed to import DP group destroy helper", exc_info=True)
        return

    try:
        stateless_destroy_torch_distributed_process_group(dp_group)
    except Exception as exc:
        errors.append(f"dp_group_destroy: {exc}")
        logger.warning("Failed to destroy DP group", exc_info=True)


class GRLLMEngine(LLMEngine):
    """LLMEngine with GR-specific input/output processors.

    After calling super().__init__(), replaces self.input_processor and
    self.output_processor so the parent's add_request() and step() work
    with GRPrompt and produce GRRequestOutput without further overrides.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
        gr_config: GRConfig | None = None,
    ) -> None:
        self.gr_config = gr_config or GRConfig()

        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
            multiprocess_mode=multiprocess_mode,
        )

        self._replace_processors()

        self._num_gr_requests_processed = 0
        self._start_time = time.time()
        self._shutdown_complete = False

        logger.info("GRLLMEngine initialized")

    def _replace_processors(self) -> None:
        """Replace default processors with GR-specific versions.

        All feature names, primary_id, and task metadata are read directly
        from the model's HSTUConfig (hf_config) -- the single source of truth.
        """
        cfg = self.gr_config
        hf_config = self.vllm_config.model_config.hf_config

        uih_feature_names = (
            hf_config.hstu_uih_feature_names
            or hf_config.user_embedding_feature_names
        )
        candidate_feature_names = (
            hf_config.hstu_candidate_feature_names
            or hf_config.item_embedding_feature_names
        )
        primary_id_feature_name = hf_config.uih_post_id_feature_name
        task_names = list(hf_config.task_names)
        primary_task = task_names[0] if task_names else ""

        self.input_processor = GRInputProcessor(
            vllm_config=self.vllm_config,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_candidates=cfg.max_candidates,
            default_top_k=cfg.default_top_k,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
            primary_id_feature_name=primary_id_feature_name,
        )

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = GROutputProcessor(
            tokenizer=self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=stream_interval,
            task_names=task_names,
            primary_task=primary_task,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: GREngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "GRLLMEngine":
        """Create GRLLMEngine from engine arguments."""
        vllm_config = engine_args.create_engine_config(usage_context)

        executor_class = Executor.get_class(vllm_config)

        gr_config = GRConfig(
            max_seq_len=engine_args.max_seq_len,
            max_candidates=engine_args.max_candidates,
            default_top_k=engine_args.default_top_k,
        )

        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
            gr_config=gr_config,
        )

    def get_gr_stats(self) -> dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "num_gr_requests_processed": self._num_gr_requests_processed,
            "elapsed_time": elapsed,
            "gr_throughput": (
                self._num_gr_requests_processed / max(elapsed, 1e-6)
            ),
            "pending_requests": self.output_processor.num_pending_requests,
        }

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return

        self._shutdown_complete = True
        logger.info(
            "GRLLMEngine shutdown: processed=%d",
            self._num_gr_requests_processed
        )

        super_shutdown = getattr(super(), "shutdown", None)
        if callable(super_shutdown):
            try:
                super_shutdown()
            except Exception:
                logger.debug("Base LLMEngine shutdown raised", exc_info=True)

        seen: set[int] = set()
        errors: list[str] = []
        for attr_name in (
            "engine_core",
            "engine_core_client",
            "model_executor",
            "executor",
        ):
            _shutdown_nested_component(
                attr_name,
                getattr(self, attr_name, None),
                seen,
                errors,
            )

        _shutdown_nested_component(
            "logger_manager",
            getattr(self, "logger_manager", None),
            seen,
            errors,
        )
        _destroy_dp_group(getattr(self, "dp_group", None), errors)

        for attr_name in (
            "executor",
            "model_executor",
            "engine_core",
            "engine_core_client",
            "logger_manager",
            "dp_group",
            "input_processor",
            "output_processor",
            "io_processor",
            "tokenizer",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        if errors:
            logger.warning(
                "GRLLMEngine shutdown completed with %d cleanup issue(s)",
                len(errors),
            )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
