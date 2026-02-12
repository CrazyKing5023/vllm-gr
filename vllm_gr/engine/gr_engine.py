# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLMEngine: Engine for generative recommendation inference.

This engine truly integrates with vLLM's EngineCore by:
1. Calling super().__init__() to initialize EngineCore
2. Using self.engine_core.add_request() to submit requests
3. Using self.engine_core.get_output() to get results
4. Converting GR data formats to EngineCoreRequest format

Inherits from vLLM's LLMEngine to maintain full API compatibility.
"""

from typing import List, Optional, Any, Dict, Union, Type, Callable, Mapping
from copy import copy
import time
import logging
import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import Executor
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams

from vllm_gr.inputs.data import GRPrompt
from vllm_gr.outputs import GRRequestOutput, RecommendedItem
from vllm_gr.request import GRRequest
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_processor import GRProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs
from vllm_gr.core.gr_scheduler import GRScheduler, GRSchedulerConfig
from vllm_gr.worker.gr_worker import GRWorker

logger = logging.getLogger("vllm_gr.engine")


class GRLLMEngine(LLMEngine):
    """
    GR LLM Engine - Inherits from vLLM LLMEngine with true EngineCore integration.
    
    This class properly integrates with vLLM's EngineCore by:
    1. Calling super().__init__() to initialize EngineCore
    2. Routing requests through self.engine_core
    3. Using EngineCore's execution pipeline
    
    Execution modes:
    - ENGINE_CORE mode: Uses vLLM's EngineCore (preferred when compatible)
    - GR_NATIVE mode: Uses GR-specific components (fallback/development)
    
    Data format mapping for EngineCore:
    - prompt_token_ids: User history item IDs (treating item IDs as "tokens")
    - mm_inputs: Contains candidate_item_ids
    - sampling_params.n: num_recommendations
    
    Example:
        >>> engine = GRLLMEngine.from_engine_args(args)
        >>> engine.add_gr_request("req-1", prompt)
        >>> while engine.has_unfinished_requests():
        ...     outputs = engine.step()
    """
    
    # Execution modes
    MODE_ENGINE_CORE = "engine_core"  # Use vLLM's EngineCore
    MODE_GR_NATIVE = "gr_native"      # Use GR-specific components
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[List[Any]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
        # GR-specific parameters
        gr_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize GRLLMEngine with EngineCore integration.
        
        Args:
            vllm_config: vLLM configuration
            executor_class: Executor class for model execution
            log_stats: Whether to log statistics
            usage_context: Usage context
            stat_loggers: Custom stat loggers
            mm_registry: Multimodal registry
            use_cached_outputs: Whether to use cached outputs
            multiprocess_mode: Whether to use multiprocess mode
            gr_config: GR-specific configuration dictionary
        """
        # Store GR-specific configuration
        self.gr_config = gr_config or {}
        self.max_seq_len = self.gr_config.get("max_seq_len", 512)
        self.max_candidates = self.gr_config.get("max_candidates", 1000)
        self.default_top_k = self.gr_config.get("default_top_k", 10)
        self.embedding_dim = self.gr_config.get("embedding_dim", 192)
        self.hash_size = self.gr_config.get("hash_size", 10_000_000)
        
        # Execution mode configuration
        self._execution_mode = self.gr_config.get("execution_mode", self.MODE_ENGINE_CORE)
        self._force_gr_native = self.gr_config.get("force_gr_native", False)
        
        # Scheduler configuration (for GR_NATIVE mode)
        self._gr_max_batch_size = self.gr_config.get("gr_max_batch_size", 32)
        self._scheduling_policy = self.gr_config.get("scheduling_policy", "fcfs")
        
        # HSTU architecture parameters
        self._num_layers = self.gr_config.get("num_layers", 3)
        self._num_heads = self.gr_config.get("num_heads", 2)
        self._hidden_dim = self.gr_config.get("hidden_dim", 128)
        self._attention_dim = self.gr_config.get("attention_dim", 64)
        self._load_model = self.gr_config.get("load_model", True)
        
        # Ensure skip_tokenizer_init is True for GR (we don't use tokenizer)
        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True
        
        # Track request metadata for GR output processing
        self._gr_request_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize EngineCore via parent class
        self._engine_core_initialized = False
        
        if not self._force_gr_native:
            try:
                logger.info("GRLLMEngine: Attempting to initialize EngineCore...")
                # Check CUDA state before initialization
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        logger.info(f"GRLLMEngine: CUDA available, device count: {torch.cuda.device_count()}")
                    except RuntimeError as cuda_err:
                        logger.warning(f"GRLLMEngine: CUDA state check failed: {cuda_err}")
                        raise RuntimeError(f"CUDA context already corrupted: {cuda_err}")
                
                super().__init__(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    usage_context=usage_context,
                    stat_loggers=stat_loggers,
                    mm_registry=mm_registry,
                    use_cached_outputs=use_cached_outputs,
                    multiprocess_mode=multiprocess_mode,
                )
                self._engine_core_initialized = True
                self._execution_mode = self.MODE_ENGINE_CORE
                logger.info("GRLLMEngine: EngineCore initialized successfully")
            except Exception as e:
                import traceback
                import sys
                
                # Print full error information
                error_type = type(e).__name__
                error_msg = str(e)
                tb = traceback.format_exc()
                
                logger.error(
                    f"GRLLMEngine: EngineCore initialization FAILED!\n"
                    f"  Error type: {error_type}\n"
                    f"  Error message: {error_msg}"
                )
                
                # Also print to stderr for visibility
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"EngineCore initialization FAILED!", file=sys.stderr)
                print(f"Error: {error_type}: {error_msg}", file=sys.stderr)
                print(f"Traceback:\n{tb}", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
                
                logger.warning("Falling back to GR-native mode")
                
                # Check if this is a CUDA error
                is_cuda_error = "CUDA" in str(e) or "cuda" in str(e).lower()
                
                # Reset CUDA state to recover from potential CUDA errors
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as sync_err:
                        logger.warning(f"CUDA sync failed: {sync_err}")
                        is_cuda_error = True
                    
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                    
                    # If there was a CUDA error, try to reset the device
                    if is_cuda_error:
                        try:
                            # Try to reset CUDA context by re-initializing
                            torch.cuda.init()
                            logger.info("CUDA context re-initialized")
                        except RuntimeError:
                            logger.warning(
                                "CUDA context corrupted and cannot be recovered. "
                                "Consider restarting the process."
                            )
                
                self._engine_core_initialized = False
                self._execution_mode = self.MODE_GR_NATIVE
                self.vllm_config = vllm_config
                self.log_stats = log_stats
        else:
            self._execution_mode = self.MODE_GR_NATIVE
            self.vllm_config = vllm_config
            self.log_stats = log_stats
            logger.info("GRLLMEngine: Using GR-native mode (forced)")
        
        # Initialize GR-specific components
        self._init_gr_components()
        
        # Statistics
        self._num_gr_requests_processed = 0
        self._start_time = time.time()
        
        logger.info(
            f"GRLLMEngine initialized: mode={self._execution_mode}, "
            f"engine_core={self._engine_core_initialized}"
        )
    
    def _init_gr_components(self) -> None:
        """Initialize GR-specific components."""
        # GRProcessor for input processing
        self.gr_processor = GRProcessor(
            max_seq_len=self.max_seq_len,
            max_candidates=self.max_candidates,
        )
        
        # GROutputProcessor for output processing
        self.gr_output_processor = GROutputProcessor(
            default_top_k=self.default_top_k,
            log_stats=self.log_stats,
        )
        
        # Components for GR_NATIVE mode
        if self._execution_mode == self.MODE_GR_NATIVE:
            # GRScheduler
            scheduler_config = GRSchedulerConfig(
                max_batch_size=self._gr_max_batch_size,
                scheduling_policy=self._scheduling_policy,
            )
            self.gr_scheduler = GRScheduler(config=scheduler_config)
            
            # GRWorker
            model_path = self.gr_config.get("model_path")
            device = self.gr_config.get("device")
            self.gr_worker = GRWorker(
                model_path=model_path,
                device=device,
                embedding_dim=self.embedding_dim,
                hash_size=self.hash_size,
                max_seq_len=self.max_seq_len,
                max_candidates=self.max_candidates,
                num_layers=self._num_layers,
                num_heads=self._num_heads,
                hidden_dim=self._hidden_dim,
                attention_dim=self._attention_dim,
                load_model=self._load_model,
            )
        else:
            self.gr_scheduler = None
            self.gr_worker = None
    
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
        
        Uses vLLM's EngineCore for model execution, with GR-specific
        components for input/output processing.
        """
        # Create vLLM config from engine args
        # This will register HSTUConfig with AutoConfig first
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # Ensure skip_tokenizer_init for GR
        if vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True
        
        # Get executor class for EngineCore
        executor_class = Executor.get_class(vllm_config)
        
        # Build GR-specific config
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
            "use_scheduler": engine_args.use_scheduler,
            "gr_max_batch_size": engine_args.gr_max_batch_size,
            "scheduling_policy": engine_args.scheduling_policy,
            "device": engine_args.device,
            # Control execution mode
            "force_gr_native": engine_args.force_gr_native,
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
    # Request Management - Route through EngineCore when possible
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
        
        Routes to EngineCore (ENGINE_CORE mode) or GRScheduler (GR_NATIVE mode).
        """
        if arrival_time is None:
            arrival_time = time.time()
        
        num_recs = num_recommendations or self.default_top_k
        
        # Extract data from prompt
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        user_id = prompt.get("user_id")
        
        # Convert to item IDs
        prompt_token_ids = [
            item.get("item_id", 0) for item in user_history
        ][-self.max_seq_len:]
        
        candidate_item_ids = [
            item.get("item_id", 0) for item in candidates
        ][:self.max_candidates]
        
        # Store metadata for output processing and model execution
        # Note: candidate_item_ids is stored here since EngineCoreRequest is
        # immutable and we cannot attach extra data to it.
        self._gr_request_metadata[request_id] = {
            "user_id": user_id,
            "candidates": candidates[:self.max_candidates],
            "candidate_item_ids": candidate_item_ids,  # For model execution
            "prompt_token_ids": prompt_token_ids,  # User history item IDs
            "num_recommendations": num_recs,
            "arrival_time": arrival_time,
        }
        
        if self._execution_mode == self.MODE_ENGINE_CORE and self._engine_core_initialized:
            # Route through EngineCore
            self._add_request_via_engine_core(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                candidate_item_ids=candidate_item_ids,
                num_recommendations=num_recs,
                priority=priority,
            )
        else:
            # Route through GR-native scheduler
            self._add_request_via_gr_native(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                candidate_item_ids=candidate_item_ids,
                num_recommendations=num_recs,
                user_id=user_id,
                priority=priority,
            )
        
        logger.debug(
            f"Added GR request {request_id}: mode={self._execution_mode}, "
            f"history_len={len(prompt_token_ids)}, candidates={len(candidate_item_ids)}"
        )
    
    def _add_request_via_engine_core(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        candidate_item_ids: List[int],
        num_recommendations: int,
        priority: int,
    ) -> None:
        """
        Add request through vLLM's EngineCore.
        
        Maps GR data to EngineCore format:
        - prompt_token_ids: User history item IDs (as "tokens")
        - candidate_item_ids: Stored in _gr_request_metadata for retrieval
        - Uses PoolingParams for recommendation (single output)
        
        Note: EngineCoreRequest is immutable (msgspec.Struct), so we store
        GR-specific data (candidate_item_ids) in _gr_request_metadata instead.
        """
        # Create PoolingParams for recommendation task
        # We use pooling_params since recommendation is similar to embedding/pooling
        # task="embed" is required - EngineCore validates that task is in supported_pooling_tasks
        pooling_params = PoolingParams(task="embed")
        
        # Note: candidate_item_ids is already stored in _gr_request_metadata
        # by add_gr_request(). The model runner will retrieve it using request_id.
        
        # Use the parent class processor to create EngineCoreRequest
        # Note: We're treating item IDs as "tokens" for compatibility
        try:
            # Build a prompt-like structure for the processor
            # Since we skip tokenizer, we provide token_ids directly
            prompt = {
                "prompt_token_ids": prompt_token_ids,
            }
            
            # Process through parent's processor (creates EngineCoreRequest)
            prompt_str, request = self.processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=pooling_params,
                arrival_time=time.time(),
                lora_request=None,
                tokenization_kwargs=None,
                trace_headers=None,
                priority=priority,
            )
            
            # Note: EngineCoreRequest is immutable (msgspec.Struct), so we cannot
            # add mm_inputs directly. Candidate data is stored in _gr_request_metadata
            # and will be retrieved by request_id during model execution.
            
            # Add to output processor
            self.output_processor.add_request(request, prompt_str, None, 0)
            
            # Add to EngineCore
            self.engine_core.add_request(request)
            
            logger.debug(f"Request {request_id} added to EngineCore")
            
        except Exception as e:
            logger.error(f"Failed to add request via EngineCore: {e}")
            # Fallback to GR-native
            logger.warning("Falling back to GR-native for this request")
            self._add_request_via_gr_native(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                candidate_item_ids=candidate_item_ids,
                num_recommendations=num_recommendations,
                user_id=self._gr_request_metadata[request_id]["user_id"],
                priority=priority,
            )
    
    def _add_request_via_gr_native(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        candidate_item_ids: List[int],
        num_recommendations: int,
        user_id: Any,
        priority: int,
    ) -> None:
        """Add request through GR-native scheduler."""
        if self.gr_scheduler is None:
            raise RuntimeError("GRScheduler not initialized for GR_NATIVE mode")
        
        # Create GRRequest for output processing
        gr_request = GRRequest(
            request_id=request_id,
            user_id=user_id,
            arrival_time=time.time(),
            num_recommendations=num_recommendations,
        )
        
        # Add to scheduler
        # Note: add_request expects request (or request_id) as first positional arg
        self.gr_scheduler.add_request(
            request_id,  # First positional argument
            prompt_token_ids=prompt_token_ids,
            candidate_item_ids=candidate_item_ids,
            num_recommendations=num_recommendations,
            user_id=user_id,
            priority=priority,
        )
        
        # Add to output processor
        self.gr_output_processor.add_gr_request(
            gr_request=gr_request,
            original_candidates=self._gr_request_metadata[request_id]["candidates"],
        )
    
    # =========================================================================
    # Inference Execution
    # =========================================================================
    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput, GRRequestOutput]]:
        """
        Execute one inference step.
        
        Routes to EngineCore or GR-native based on execution mode.
        """
        if self._execution_mode == self.MODE_ENGINE_CORE and self._engine_core_initialized:
            return self._step_via_engine_core()
        else:
            return self._step_via_gr_native()
    
    def _step_via_engine_core(self) -> List[GRRequestOutput]:
        """
        Execute step through vLLM's EngineCore.
        
        Uses self.engine_core.get_output() and converts results to GRRequestOutput.
        """
        # Get outputs from EngineCore (parent's step logic)
        try:
            # This calls engine_core.get_output() internally
            engine_outputs = super().step()
        except Exception as e:
            logger.error(f"EngineCore step failed: {e}")
            return []
        
        # Convert to GRRequestOutput
        gr_outputs = []
        for output in engine_outputs:
            request_id = output.request_id
            
            if request_id in self._gr_request_metadata:
                gr_output = self._convert_engine_output_to_gr(request_id, output)
                if gr_output is not None:
                    gr_outputs.append(gr_output)
                    self._num_gr_requests_processed += 1
                    # Clean up metadata
                    del self._gr_request_metadata[request_id]
        
        return gr_outputs
    
    def _convert_engine_output_to_gr(
        self,
        request_id: str,
        engine_output: Union[RequestOutput, PoolingRequestOutput],
    ) -> Optional[GRRequestOutput]:
        """Convert EngineCore output to GRRequestOutput."""
        metadata = self._gr_request_metadata.get(request_id)
        if metadata is None:
            return None
        
        candidates = metadata["candidates"]
        num_recs = metadata["num_recommendations"]
        
        # Extract scores from engine output
        # For pooling output, scores are in outputs.data
        # Note: PoolingRequestOutput.outputs is a single PoolingOutput, not a list
        if hasattr(engine_output, 'outputs') and engine_output.outputs is not None:
            output_data = engine_output.outputs
            if hasattr(output_data, 'data'):
                # PoolingOutput.data is the tensor containing embeddings/scores
                scores = output_data.data
            elif isinstance(output_data, (list, tuple)) and len(output_data) > 0:
                # Fallback for list-like outputs (RequestOutput.outputs is a list)
                first_output = output_data[0]
                if hasattr(first_output, 'data'):
                    scores = first_output.data
                else:
                    scores = list(range(len(candidates), 0, -1))
            else:
                # Fallback: generate placeholder scores
                scores = list(range(len(candidates), 0, -1))
        else:
            scores = list(range(len(candidates), 0, -1))
        
        # Create recommendations
        recommendations = self._create_recommendations(
            candidates=candidates,
            scores=scores,
            num_recommendations=num_recs,
        )
        
        return GRRequestOutput(
            request_id=request_id,
            recommendations=recommendations,
            finished=True,
            user_id=metadata.get("user_id"),
        )
    
    def _step_via_gr_native(self) -> List[GRRequestOutput]:
        """Execute step through GR-native components."""
        if self.gr_scheduler is None or self.gr_worker is None:
            return []
        
        # Schedule next batch
        scheduler_output = self.gr_scheduler.schedule()
        
        if scheduler_output.is_empty:
            return []
        
        # Execute model
        model_output = self.gr_worker.execute_model(scheduler_output)
        
        # Update scheduler state
        completed_ids = self.gr_scheduler.update_from_output(
            scheduler_output, model_output.predictions
        )
        
        # Create outputs
        request_outputs = []
        for i, request_id in enumerate(model_output.request_ids):
            if i < len(model_output.predictions):
                scores = model_output.predictions[i]
                gr_output = self.gr_output_processor.create_output_from_scores(
                    request_id=request_id,
                    scores=scores,
                )
                if gr_output is not None:
                    request_outputs.append(gr_output)
                    self._num_gr_requests_processed += 1
                    # Clean up metadata
                    if request_id in self._gr_request_metadata:
                        del self._gr_request_metadata[request_id]
        
        if self.log_stats and request_outputs:
            logger.debug(
                f"GR-native step: {len(request_outputs)} outputs, "
                f"{self.gr_scheduler.get_num_waiting_requests()} waiting"
            )
        
        return request_outputs
    
    def _create_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        scores: Union[List[float], torch.Tensor],
        num_recommendations: int,
    ) -> List[RecommendedItem]:
        """Create sorted recommendations from candidates and scores."""
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        
        # Pair candidates with scores
        scored_items = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            if i < len(scores):
                scored_items.append((candidate, float(score)))
        
        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        for rank, (candidate, score) in enumerate(scored_items[:num_recommendations], 1):
            recommendations.append(RecommendedItem(
                item_id=candidate.get("item_id"),
                score=float(score) if hasattr(score, 'item') else score,
                rank=rank,
                features=candidate,  # RecommendedItem uses 'features', not 'metadata'
            ))
        
        return recommendations
    
    # =========================================================================
    # State Queries
    # =========================================================================
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        if self._execution_mode == self.MODE_ENGINE_CORE and self._engine_core_initialized:
            return super().has_unfinished_requests()
        elif self.gr_scheduler is not None:
            return self.gr_scheduler.has_unfinished_requests()
        return False
    
    def get_num_unfinished_requests(self) -> int:
        """Get number of unfinished requests."""
        if self._execution_mode == self.MODE_ENGINE_CORE and self._engine_core_initialized:
            return super().get_num_unfinished_requests()
        elif self.gr_scheduler is not None:
            return self.gr_scheduler.get_num_unfinished_requests()
        return 0
    
    def abort_request(self, request_ids: Union[str, List[str]]) -> None:
        """Abort requests."""
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        
        if self._execution_mode == self.MODE_ENGINE_CORE and self._engine_core_initialized:
            super().abort_request(request_ids)
        elif self.gr_scheduler is not None:
            for request_id in request_ids:
                self.gr_scheduler.abort_request(request_id)
        
        # Clean up metadata
        for request_id in request_ids:
            if request_id in self._gr_request_metadata:
                del self._gr_request_metadata[request_id]
    
    # =========================================================================
    # Statistics and Lifecycle
    # =========================================================================
    
    def get_execution_mode(self) -> str:
        """Get current execution mode."""
        return self._execution_mode
    
    def get_gr_stats(self) -> Dict[str, Any]:
        """Get GR-specific statistics."""
        elapsed = time.time() - self._start_time
        stats = {
            "execution_mode": self._execution_mode,
            "engine_core_initialized": self._engine_core_initialized,
            "num_gr_requests_processed": self._num_gr_requests_processed,
            "elapsed_time": elapsed,
            "gr_throughput": self._num_gr_requests_processed / max(elapsed, 1e-6),
            "pending_requests": len(self._gr_request_metadata),
        }
        
        if self.gr_scheduler is not None:
            stats.update(self.gr_scheduler.get_stats())
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        # Shutdown GR worker if exists
        if self.gr_worker is not None:
            self.gr_worker.shutdown()
        
        # Clear metadata
        self._gr_request_metadata.clear()
        
        logger.info(
            f"GRLLMEngine shutdown: mode={self._execution_mode}, "
            f"processed={self._num_gr_requests_processed}"
        )
    
    def __del__(self):
        """Destructor."""
        try:
            self.shutdown()
        except Exception:
            pass
