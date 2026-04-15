# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLMEngine — extends vLLM's LLMEngine for recommendation inference.

After standard initialization, replaces the default input/output processors
with GR-specific versions (GRInputProcessor, GROutputProcessor) so the
existing add_request() / step() pipeline handles GRPrompt transparently.
"""

import gc
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
from vllm_gr.utils.npu_adapter import release_accelerator_memory

logger = logging.getLogger("vllm_gr.engine")


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

        logger.info("GRLLMEngine initialized")

    def get_prefix_cache_counters(self) -> dict[str, int]:
        getter = getattr(self.output_processor, "get_prefix_cache_counters", None)
        if callable(getter):
            return getter()
        return {"requests": 0, "queries": 0, "hits": 0}

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True

        for attr_name in ("executor", "model_executor"):
            executor = getattr(self, attr_name, None)
            if executor is None:
                continue

            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close = getattr(executor, "close", None)
                if callable(close):
                    close()

            setattr(self, attr_name, None)

        engine_core = getattr(self, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()
            self.engine_core = None

        for attr_name in (
            "input_processor",
            "output_processor",
            "tokenizer",
            "detokenizer",
            "logger_manager",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        vllm_config = getattr(self, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        static_forward_context = getattr(
            compilation_config,
            "static_forward_context",
            None,
        )
        if isinstance(static_forward_context, dict):
            static_forward_context.clear()
        try:
            from vllm.config import get_cached_compilation_config
        except ImportError:
            get_cached_compilation_config = None
        if get_cached_compilation_config is not None:
            get_cached_compilation_config.cache_clear()

        gc.collect()
        release_accelerator_memory()
        logger.info("GRLLMEngine shutdown complete")

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
        configured_context_feature_names = list(dict.fromkeys(
            list(getattr(hf_config, "contextual_feature_to_max_length", {}).keys())
            + list(getattr(hf_config, "contextual_feature_to_min_uih_length", {}).keys())
        ))
        context_feature_names = [
            key for key in uih_feature_names
            if key in set(configured_context_feature_names)
        ]
        primary_id_feature_name = hf_config.uih_post_id_feature_name
        candidate_primary_id_key = self._resolve_candidate_primary_id_key(
            hf_config=hf_config,
            candidate_feature_names=candidate_feature_names,
        )
        task_names = list(hf_config.task_names)
        primary_task = task_names[0] if task_names else ""

        self.input_processor = GRInputProcessor(
            vllm_config=self.vllm_config,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_candidates=cfg.max_candidates,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
            primary_id_feature_name=primary_id_feature_name,
            full_uih_feature_names=uih_feature_names,
            context_feature_names=context_feature_names,
        )

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = GROutputProcessor(
            tokenizer=self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=stream_interval,
            task_names=task_names,
            primary_task=primary_task,
            primary_id_key=candidate_primary_id_key,
        )

    @staticmethod
    def _resolve_candidate_primary_id_key(
        *,
        hf_config,
        candidate_feature_names: list[str],
    ) -> str:
        explicit_candidate_id_key = str(
            getattr(hf_config, "candidate_primary_id_feature_name", "") or ""
        )
        if explicit_candidate_id_key:
            return explicit_candidate_id_key

        merge_mapping = list(
            getattr(hf_config, "merge_uih_candidate_feature_mapping", []) or []
        )
        primary_uih_id_key = getattr(hf_config, "uih_post_id_feature_name", "")
        for uih_key, candidate_key in merge_mapping:
            if uih_key == primary_uih_id_key:
                return candidate_key

        raise ValueError(
            "Unable to resolve candidate primary id feature. Configure "
            "`candidate_primary_id_feature_name` explicitly or provide a "
            "`merge_uih_candidate_feature_mapping` entry that maps "
            f"{primary_uih_id_key!r} to one of {candidate_feature_names!r}."
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
