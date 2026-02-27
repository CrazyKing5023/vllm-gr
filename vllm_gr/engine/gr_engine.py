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

from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor

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
        gr_config: dict[str, Any] | None = None,
    ) -> None:
        self.gr_config = gr_config or {}

        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True

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

        logger.info("GRLLMEngine initialized")

    def _replace_processors(self) -> None:
        """Replace default processors with GR-specific versions."""
        max_seq_len = self.gr_config.get("max_seq_len", 512)
        max_candidates = self.gr_config.get("max_candidates", 1000)
        default_top_k = self.gr_config.get("default_top_k", 10)
        uih_feature_names = self.gr_config.get("uih_feature_names", None)
        candidate_feature_names = self.gr_config.get(
            "candidate_feature_names", None)

        self.input_processor = GRInputProcessor(
            vllm_config=self.vllm_config,
            tokenizer=self.tokenizer,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
        )

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = GROutputProcessor(
            tokenizer=self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=stream_interval,
            default_top_k=default_top_k,
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

        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True

        executor_class = Executor.get_class(vllm_config)

        gr_config = {
            "model_path": engine_args.model if engine_args.model else None,
            "max_seq_len": engine_args.max_seq_len,
            "max_candidates": engine_args.max_candidates,
            "default_top_k": engine_args.default_top_k,
            "embedding_dim": engine_args.embedding_dim,
            "hash_size": engine_args.hash_size,
            "load_model": engine_args.load_model,
            "num_layers": engine_args.num_layers,
            "num_heads": engine_args.num_heads,
            "hidden_dim": engine_args.hidden_dim,
            "attention_dim": engine_args.attention_dim,
            "device": engine_args.device,
            "uih_feature_names": engine_args.uih_feature_names,
            "candidate_feature_names": engine_args.candidate_feature_names,
        }

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
            "pending_requests": self.output_processor.get_num_unfinished_requests(),
        }

    def shutdown(self) -> None:
        logger.info(
            "GRLLMEngine shutdown: processed=%d",
            self._num_gr_requests_processed
        )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
