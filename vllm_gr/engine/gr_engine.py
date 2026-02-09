# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLMEngine: LLM Engine for generative recommendation.

Inherits from vLLM LLMEngine and customizes for recommendation tasks.
"""

from typing import Any, Dict, List, Mapping, Optional, Union
import time
import logging

# Required imports - these are mandatory dependencies
import torch

logger = logging.getLogger("vllm_gr.engine")

# vLLM imports - mandatory dependency (vLLM 0.11.0)
# Try v1 API first, fall back to v0
try:
    from vllm.v1.engine.llm_engine import LLMEngine
    _VLLM_VERSION = "v1"
except ImportError:
    from vllm.engine.llm_engine import LLMEngine
    _VLLM_VERSION = "v0"

from vllm_gr.inputs.data import GRPrompt, GRInputs
from vllm_gr.outputs import GRRequestOutput
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_processor import GRProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor
from vllm_gr.engine.model_runner import GRModelRunner


class GRLLMEngine(LLMEngine):
    """
    Generative Recommendation Engine.
    
    Inherits from vLLM LLMEngine and extends for recommendation tasks.
    
    Similarities with LLMEngine:
    - Uses causal attention - autoregressive architecture
    - Transformer architecture (HSTU is a Transformer variant)
    
    Inheritance strategy:
    1. Reuse LLMEngine's factory methods, config management, logging, distributed support
    2. Replace Processor with GRProcessor (handles GRPrompt)
    3. Replace OutputProcessor with GROutputProcessor (generates GRRequestOutput)
    4. Add recommendation-specific methods (add_gr_request, recommend)
    
    Key differences from LLMEngine:
    - step() semantics differ: LLM generates one token, GR uses Target-Aware Attention
      to evaluate all candidates in parallel, one step() completes the entire request
    - Input is GRPrompt (user history + candidates) instead of PromptType (token sequence)
    - Output is GRRequestOutput (candidate ranking scores) instead of RequestOutput (token sequence)
    """
    
    def __init__(
        self,
        vllm_config: Optional[Any] = None,
        executor_class: Optional[type] = None,
        log_stats: bool = True,
        usage_context: Optional[Any] = None,
        stat_loggers: Optional[list] = None,
        multiprocess_mode: bool = False,
        # GR-specific parameters
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        default_top_k: int = 10,
        uih_feature_names: Optional[List[str]] = None,
        candidate_feature_names: Optional[List[str]] = None,
        # Model loading parameters
        model_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        load_model: bool = True,
        # HSTU model architecture parameters (used when creating model from config without pretrained weights)
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
    ) -> None:
        """
        Initialize GRLLMEngine.
        
        Args:
            vllm_config: vLLM configuration
            executor_class: Executor class
            log_stats: Whether to log statistics
            usage_context: Usage context
            stat_loggers: Statistics loggers
            multiprocess_mode: Whether to enable multiprocess mode
            max_seq_len: Maximum sequence length
            max_candidates: Maximum number of candidates
            default_top_k: Default Top-K value
            uih_feature_names: User history feature names
            candidate_feature_names: Candidate feature names
            model_path: Model path (for loading HSTU model)
            embedding_path: Embedding table path (optional)
            device: Inference device (cuda/cpu)
            embedding_dim: Embedding dimension
            hash_size: Hash table size
            load_model: Whether to load and initialize the HSTU model
            num_layers: Number of HSTU layers
            num_heads: Number of attention heads
            hidden_dim: Hidden layer dimension
            attention_dim: Attention dimension
        """
        # Save GR-specific configuration
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.default_top_k = default_top_k
        self.log_stats = log_stats
        self.vllm_config = vllm_config
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.device = device
        
        # Call parent initialization (if valid vllm_config and executor_class exist)
        if vllm_config is not None and executor_class is not None:
            try:
                super().__init__(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    usage_context=usage_context,
                    stat_loggers=stat_loggers,
                    multiprocess_mode=multiprocess_mode,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize vLLM LLMEngine: {e}")
        
        # Create GR-specific processor
        self.processor = GRProcessor(
            vllm_config=vllm_config,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
        )
        
        # Create GR output processor
        self.output_processor = GROutputProcessor(
            default_top_k=default_top_k,
            log_stats=log_stats,
        )
        
        # Request queue
        self._pending_requests: Dict[str, Any] = {}
        
        # Create model runner (for actual inference)
        self._model_runner: Optional[GRModelRunner] = None
        if load_model:
            try:
                self._model_runner = GRModelRunner(
                    model_path=model_path,
                    embedding_path=embedding_path,
                    device=device,
                    max_seq_len=max_seq_len,
                    max_candidates=max_candidates,
                    embedding_dim=embedding_dim,
                    hash_size=hash_size,
                    # HSTU model architecture parameters
                    num_layers=num_layers,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    attention_dim=attention_dim,
                )
            except Exception as e:
                logger.warning(f"Failed to create GRModelRunner: {e}")
                self._model_runner = None
        
        # Compatibility: retain _model attribute
        self._model = None
        
        logger.info(
            f"GRLLMEngine initialized: max_seq_len={max_seq_len}, "
            f"max_candidates={max_candidates}, default_top_k={default_top_k}, "
            f"vLLM version={_VLLM_VERSION}, model_runner={'enabled' if self._model_runner else 'disabled'}"
        )
    
    @classmethod
    def from_engine_args(
        cls,
        engine_args: Union[GREngineArgs, Any],
        usage_context: Optional[Any] = None,
        stat_loggers: Optional[list] = None,
        enable_multiprocessing: bool = False,
    ) -> "GRLLMEngine":
        """
        Create instance from engine arguments (overrides parent method).
        
        Args:
            engine_args: Engine arguments (GREngineArgs or vLLM EngineArgs)
            usage_context: Usage context
            stat_loggers: Statistics loggers
            enable_multiprocessing: Whether to enable multiprocessing
            
        Returns:
            GRLLMEngine instance
        """
        # Extract GR-specific parameters
        gr_kwargs = {
            "max_seq_len": getattr(engine_args, 'max_seq_len', 512),
            "max_candidates": getattr(engine_args, 'max_candidates', 1000),
            "default_top_k": getattr(engine_args, 'default_top_k', 10),
            "uih_feature_names": getattr(engine_args, 'uih_feature_names', None),
            "candidate_feature_names": getattr(engine_args, 'candidate_feature_names', None),
            # Model loading parameters
            "model_path": getattr(engine_args, 'model', None),
            "embedding_path": getattr(engine_args, 'embedding_path', None),
            "device": getattr(engine_args, 'device', None),
            "embedding_dim": getattr(engine_args, 'embedding_dim', 192),
            "hash_size": getattr(engine_args, 'hash_size', 10_000_000),
            "load_model": getattr(engine_args, 'load_model', True),
            # HSTU model architecture parameters
            "num_layers": getattr(engine_args, 'num_layers', 3),
            "num_heads": getattr(engine_args, 'num_heads', 2),
            "hidden_dim": getattr(engine_args, 'hidden_dim', 128),
            "attention_dim": getattr(engine_args, 'attention_dim', 64),
        }
        
        vllm_config = None
        executor_class = None
        log_stats = not getattr(engine_args, 'disable_log_stats', False)
        use_vllm_registry = getattr(engine_args, 'use_vllm_registry', False)
        
        # HSTU Model Loading Strategy:
        #
        # Option 1: vLLM ModelRegistry (use_vllm_registry=True)
        # - HSTU is registered with vLLM's ModelRegistry via register_hstu_model()
        # - Uses standard vLLM model loading pipeline
        # - Requires: model_type="hstu", architectures=["HSTUForRecommendation"]
        #   in config.json
        #
        # Option 2: GR-native loading (default, use_vllm_registry=False)
        # - Uses GR's own model loading: HSTUModelConfig + GRModelRunner
        # - More flexible for custom HSTU configurations
        # - Does not require HuggingFace-compatible config.json
        
        if use_vllm_registry:
            # Try to use vLLM's standard model loading with registered HSTU
            try:
                from vllm_gr.registry import is_hstu_registered, register_hstu_model
                
                if not is_hstu_registered():
                    register_hstu_model()
                
                if hasattr(engine_args, 'create_engine_config'):
                    vllm_config = engine_args.create_engine_config(usage_context)
                    logger.info("Using vLLM ModelRegistry for HSTU model loading")
            except Exception as e:
                logger.warning(f"Failed to use vLLM ModelRegistry: {e}")
                logger.info("Falling back to GR-native model loading")
                vllm_config = None
        else:
            logger.debug(
                "GRLLMEngine: Using GR-native model loading "
                "(set use_vllm_registry=True to use vLLM ModelRegistry)"
            )
        
        # Create GRLLMEngine
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
            **gr_kwargs,
        )
    
    # ==================== Recommendation Interface ====================
    
    def add_gr_request(
        self,
        request_id: str,
        prompt: GRPrompt,
        num_recommendations: Optional[int] = None,
        arrival_time: Optional[float] = None,
        priority: int = 0,
    ) -> None:
        """
        Add recommendation request (GR-specific method).
        
        Differences from parent add_request:
        - Receives GRPrompt instead of PromptType
        - Uses num_recommendations instead of SamplingParams
        
        Args:
            request_id: Unique request identifier
            prompt: Recommendation request in GRPrompt format
            num_recommendations: Number of recommendations to return (defaults to default_top_k)
            arrival_time: Request arrival time
            priority: Request priority
        """
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")
        
        if arrival_time is None:
            arrival_time = time.time()
        
        num_recs = num_recommendations or self.default_top_k
        
        # Process input using GRProcessor
        gr_request = self.processor.process_gr_inputs(
            request_id=request_id,
            prompt=prompt,
            num_recommendations=num_recs,
            arrival_time=arrival_time,
            priority=priority,
        )
        
        # Add to OutputProcessor for tracking
        self.output_processor.add_gr_request(
            gr_request,
            original_candidates=prompt.get("candidates", []),
        )
        
        # Add to internal request queue
        self._pending_requests[request_id] = {
            "request": gr_request,
            "prompt": prompt,
            "arrival_time": arrival_time,
        }
        
        # Add to engine_core if available
        if hasattr(self, 'engine_core') and self.engine_core is not None:
            try:
                self.engine_core.add_request(gr_request)
            except Exception as e:
                logger.warning(f"Failed to add request to engine_core: {e}")
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return (
            self.output_processor.has_unfinished_requests() or 
            len(self._pending_requests) > 0
        )
    
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return max(
            self.output_processor.get_num_unfinished_requests(),
            len(self._pending_requests),
        )
    
    # ==================== Step Method Override ====================
    
    def step(self) -> List[GRRequestOutput]:
        """
        Execute one inference step (overrides parent method).
        
        Key differences from LLM step():
        - LLM: Generates one token, may need multiple step() calls to complete a request
        - GR: Uses Target-Aware Attention to evaluate all candidates in parallel
              One step() completes the entire request (returns finished=True)
        
        Returns:
            List[GRRequestOutput]: Completed recommendation request outputs
        """
        outputs = []
        
        # Get output from engine_core if available
        if hasattr(self, 'engine_core') and self.engine_core is not None:
            try:
                engine_outputs = self.engine_core.get_output()
                processed = self.output_processor.process_gr_outputs(engine_outputs)
                outputs.extend(processed.request_outputs)
            except Exception as e:
                logger.warning(f"Failed to get output from engine_core: {e}")
        
        # Process pending requests
        if self._pending_requests:
            outputs.extend(self._process_pending_requests())
        
        # Clean up completed requests
        completed_ids = [o.request_id for o in outputs if o.finished]
        for req_id in completed_ids:
            if req_id in self._pending_requests:
                del self._pending_requests[req_id]
        
        return outputs
    
    def _process_pending_requests(self) -> List[GRRequestOutput]:
        """Process pending requests (internal method)."""
        outputs = []
        
        for request_id, req_info in list(self._pending_requests.items()):
            gr_request = req_info["request"]
            prompt = req_info["prompt"]
            
            scores = None
            
            # Use GRModelRunner for inference
            if self._model_runner is not None:
                try:
                    scores = self._model_runner.run_inference(gr_request, prompt)
                except Exception as e:
                    logger.error(f"GRModelRunner inference failed: {e}")
                    raise RuntimeError(f"Model inference failed: {e}") from e
            
            # Fallback: use legacy _model interface (for compatibility)
            elif self._model is not None and hasattr(self._model, 'forward'):
                try:
                    scores = self._model.forward(gr_request.gr_inputs)
                except Exception as e:
                    logger.error(f"Model forward failed: {e}")
                    raise RuntimeError(f"Model forward failed: {e}") from e
            
            # No model available - raise error
            else:
                raise RuntimeError(
                    "No model available for inference. "
                    "Please provide a valid model path or set load_model=True."
                )
            
            if scores is not None:
                output = self.output_processor.create_output_from_scores(
                    request_id=request_id,
                    scores=scores,
                )
                if output is not None:
                    outputs.append(output)
        
        return outputs
    
    def set_model(self, model: Any) -> None:
        """Set model (for direct inference)."""
        self._model = model
    
    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        if request_id in self._pending_requests:
            del self._pending_requests[request_id]
        self.output_processor.remove_request(request_id)
        
        # Also remove from engine_core if available
        if hasattr(self, 'engine_core') and self.engine_core is not None:
            try:
                self.engine_core.abort_request(request_id)
            except Exception:
                pass
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        self._pending_requests.clear()
        
        # Shutdown model runner
        if self._model_runner is not None:
            try:
                self._model_runner.shutdown()
            except Exception:
                pass
            self._model_runner = None
        
        # Call parent's shutdown method
        if hasattr(super(), 'shutdown'):
            try:
                super().shutdown()
            except Exception:
                pass
        
        logger.info("GRLLMEngine shutdown complete")
