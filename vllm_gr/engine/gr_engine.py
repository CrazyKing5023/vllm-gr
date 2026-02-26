# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLMEngine: Engine for generative recommendation inference.

Inherits from vLLM's LLMEngine and integrates with EngineCore by:
1. Using GRInputProcessor to convert GRPrompt → GREngineCoreRequest
   with full GRFeaturesPayload (not just item_ids).
2. Submitting GREngineCoreRequest directly to engine_core.
3. Using GROutputProcessor to convert engine outputs → GRRequestOutput.

The monkey-patched types (GREngineCoreRequest, GRRequest, GRNewRequestData)
ensure that gr_features flows through the entire pipeline:
  GRInputProcessor → EngineCore → Scheduler → GRModelRunner → HSTU model
"""

from typing import List, Optional, Any, Dict, Union, Type
import time
import logging

import torch

from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import Executor
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput, PoolingRequestOutput

from vllm_gr.inputs.data import GRPrompt
from vllm_gr.outputs import GRRequestOutput, RecommendedItem
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs
from vllm_gr.patch import re_apply_patches

logger = logging.getLogger("vllm_gr.engine")


class GRLLMEngine(LLMEngine):
    """
    GR LLM Engine — Inherits from vLLM LLMEngine with EngineCore integration.

    Data flow:
        GRPrompt
         → GRInputProcessor.process_gr_prompt()
         → GREngineCoreRequest (with gr_features payload)
         → engine_core.add_request()
         → [Scheduler creates GRNewRequestData with gr_features]
         → [GRModelRunner._update_states decodes gr_features]
         → [GRModelRunner._model_forward injects embeddings]
         → [HSTU model forward]
         → engine_core.get_output()
         → GROutputProcessor.process_outputs()
         → GRRequestOutput (Top-K recommendations)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[List[Any]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
        # GR-specific parameters
        gr_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Store GR-specific configuration
        self.gr_config = gr_config or {}
        self.max_seq_len = self.gr_config.get("max_seq_len", 512)
        self.max_candidates = self.gr_config.get("max_candidates", 1000)
        self.default_top_k = self.gr_config.get("default_top_k", 10)

        if multiprocess_mode:
            logger.warning(
                "GRLLMEngine: multiprocess_mode=True is experimental. "
                "GR features may not survive ZMQ serialization unless "
                "vllm_gr is imported in the engine subprocess. "
                "Prefer multiprocess_mode=False (default) for GR."
            )

        # Re-apply monkey patches to cover late-loaded vLLM modules
        # (e.g. vllm.v1.engine.core, vllm.v1.engine.core_client).
        re_apply_patches()

        # Ensure skip_tokenizer_init is True for GR (we don't use tokenizer)
        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True

        # Initialize EngineCore via parent class (v0.14.1 signature)
        logger.info("GRLLMEngine: Initializing EngineCore...")
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
        logger.info("GRLLMEngine: EngineCore initialized successfully")

        # Initialize GR-specific components
        self._init_gr_components()

        # Statistics
        self._num_gr_requests_processed = 0
        self._start_time = time.time()

        logger.info("GRLLMEngine initialized with EngineCore mode")

    def _init_gr_components(self) -> None:
        """Initialize GR-specific input/output processors."""
        uih_feature_names = self.gr_config.get("uih_feature_names", None)
        candidate_feature_names = self.gr_config.get(
            "candidate_feature_names", None
        )

        # GRInputProcessor: GRPrompt → GREngineCoreRequest
        self.gr_input_processor = GRInputProcessor(
            max_seq_len=self.max_seq_len,
            max_candidates=self.max_candidates,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
        )

        # GROutputProcessor: engine outputs → GRRequestOutput
        self.gr_output_processor = GROutputProcessor(
            default_top_k=self.default_top_k,
            log_stats=self.log_stats,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: GREngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[List[Any]] = None,
        enable_multiprocessing: bool = False,
    ) -> "GRLLMEngine":
        """
        Create GRLLMEngine from engine arguments.

        Uses vLLM's EngineCore with GR-specific processors and
        custom GRWorker (injected via worker_cls).
        """
        # Create vLLM config (registers HSTUConfig + GR models)
        vllm_config = engine_args.create_engine_config(usage_context)

        # Ensure skip_tokenizer_init for GR
        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True

        # Get executor class
        executor_class = Executor.get_class(vllm_config)

        # Build GR config dict
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

    # =========================================================================
    # Request Management
    # =========================================================================

    def add_gr_request(
        self,
        request_id: str,
        prompt: GRPrompt,
        num_recommendations: Optional[int] = None,
        arrival_time: Optional[float] = None,
        priority: int = 0,
    ) -> None:
        """
        Add a GR recommendation request.

        Uses GRInputProcessor to convert GRPrompt into GREngineCoreRequest
        with full GRFeaturesPayload, then submits to EngineCore.
        """
        if arrival_time is None:
            arrival_time = time.time()

        num_recs = num_recommendations or self.default_top_k

        # 1. Use GRInputProcessor to create GREngineCoreRequest
        gr_request = self.gr_input_processor.process_gr_prompt(
            request_id=request_id,
            prompt=prompt,
            num_recommendations=num_recs,
            arrival_time=arrival_time,
            priority=priority,
        )

        # 2. Assign internal request_id (v0.14.1 requirement).
        #    Sets external_req_id = original request_id, then replaces
        #    request_id with a unique internal ID. OutputProcessor
        #    asserts external_req_id is not None.
        self.input_processor.assign_request_id(gr_request)

        # 3. Register with GROutputProcessor for output tracking.
        #    Use the ORIGINAL request_id (now in external_req_id)
        #    because super().step() returns outputs keyed by external_req_id.
        user_id = prompt.get("user_id")
        original_candidates = (
            gr_request.gr_features.original_candidates
            if gr_request.gr_features
            else prompt.get("candidates", [])[:self.max_candidates]
        )
        self.gr_output_processor.add_request(
            request_id=gr_request.external_req_id,
            user_id=user_id,
            num_recommendations=num_recs,
            original_candidates=original_candidates,
        )

        # 4. Register with parent's output_processor (for EngineCore tracking)
        self.output_processor.add_request(gr_request, None, None, 0)

        # 5. Submit GREngineCoreRequest to EngineCore
        self.engine_core.add_request(gr_request)

        logger.debug(
            f"Added GR request {request_id}: "
            f"history_len={len(prompt.get('user_history', []))}, "
            f"candidates={len(prompt.get('candidates', []))}, "
            f"num_recs={num_recs}"
        )

    # =========================================================================
    # Inference Execution
    # =========================================================================

    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput, GRRequestOutput]]:
        """
        Execute one inference step via EngineCore.

        Delegates to parent's step() which handles all v0.14.1 internals
        (dummy_batch for DP, IterationStats, StatLoggerManager, abort of
        stop-string requests), then converts outputs to GRRequestOutput.
        """
        try:
            engine_outputs = super().step()
        except Exception as e:
            logger.error(f"EngineCore step failed: {e}")
            return []

        if not engine_outputs:
            return []

        processed = self.gr_output_processor.process_outputs(engine_outputs)

        if processed.request_outputs:
            self._num_gr_requests_processed += len(processed.request_outputs)

        return processed.request_outputs

    # =========================================================================
    # State Queries — Delegate to EngineCore (parent)
    # =========================================================================

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return super().has_unfinished_requests()

    def get_num_unfinished_requests(self) -> int:
        """Get number of unfinished requests."""
        return super().get_num_unfinished_requests()

    def abort_request(
        self, request_ids: Union[str, List[str]], internal: bool = False
    ) -> None:
        """Abort requests (v0.14.1 signature includes internal flag)."""
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        super().abort_request(request_ids, internal=internal)

        for request_id in request_ids:
            self.gr_output_processor.remove_request(request_id)

    # =========================================================================
    # Statistics and Lifecycle
    # =========================================================================

    def get_gr_stats(self) -> Dict[str, Any]:
        """Get GR-specific statistics."""
        elapsed = time.time() - self._start_time
        return {
            "num_gr_requests_processed": self._num_gr_requests_processed,
            "elapsed_time": elapsed,
            "gr_throughput": (
                self._num_gr_requests_processed / max(elapsed, 1e-6)
            ),
            "pending_requests": (
                self.gr_output_processor.get_num_unfinished_requests()
            ),
        }

    def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info(
            f"GRLLMEngine shutdown: processed={self._num_gr_requests_processed}"
        )

    def __del__(self):
        """Destructor."""
        try:
            self.shutdown()
        except Exception:
            pass
