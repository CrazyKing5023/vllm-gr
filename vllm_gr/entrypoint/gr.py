# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLM: Generative Recommendation LLM entry point.

Inherits from vLLM's LLM class and provides a recommendation-specific interface.
"""

from typing import Any, Optional, Union, Sequence, List, Dict
import time
import logging

# Required imports - these are mandatory dependencies
from tqdm.auto import tqdm
import torch

# vLLM imports - mandatory dependency (vLLM 0.11.0)
from vllm.entrypoints.llm import LLM
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

logger = logging.getLogger("vllm_gr.entrypoint")

from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.outputs import GRRequestOutput
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_engine import GRLLMEngine


class GRLLM(LLM):
    """
    Generative Recommendation LLM - Inherits from vLLM's LLM class.
    
    Responsibility boundaries (division with GRLLMEngine):
    - GRLLM: User interface layer, handles request validation, ID generation, result collection
    - GRLLMEngine: Engine layer, handles processor management, scheduling, model inference
    
    GRLLM does not directly create or manage processors; all delegated to GRLLMEngine.
    
    Args:
        model: Model name or path (HSTU model)
        embedding_path: Embedding table path (optional)
        max_seq_len: Maximum sequence length, default 512
        max_candidates: Maximum number of candidates, default 1000
        default_top_k: Default number of recommendations, default 10
        tensor_parallel_size: Tensor parallelism degree, default 1
        gpu_memory_utilization: GPU memory utilization, default 0.9
        device: Inference device (cuda/cpu), None for auto-selection
        embedding_dim: Embedding vector dimension, default 192
        hash_size: Embedding table hash size, default 10_000_000
        load_model: Whether to load and initialize the HSTU model
        num_layers: Number of HSTU layers (used without pretrained weights)
        num_heads: Number of attention heads (used without pretrained weights)
        hidden_dim: Hidden layer dimension (used without pretrained weights)
        attention_dim: Attention dimension (used without pretrained weights)
        use_scheduler: Whether to use scheduler for batching, default True
        max_batch_size: Maximum batch size for scheduler, default 32
        scheduling_policy: Scheduling policy ('fcfs' or 'priority'), default 'fcfs'
    
    Example:
        >>> from vllm_gr import GRLLM
        >>> 
        >>> # Initialize
        >>> llm = GRLLM(
        ...     model="path/to/hstu_model",
        ...     tensor_parallel_size=2,
        ... )
        >>> 
        >>> # Build recommendation request - using numeric IDs
        >>> prompt = {
        ...     "user_id": 123,
        ...     "user_history": [
        ...         {"item_id": 1, "timestamp": 1000, "rating": 4},
        ...         {"item_id": 50, "timestamp": 2000, "rating": 5},
        ...     ],
        ...     "candidates": [
        ...         {"item_id": i} for i in range(1, 101)
        ...     ],
        ...     "num_recommendations": 10,
        ... }
        >>> 
        >>> # Get recommendation results
        >>> outputs = llm.recommend([prompt])
        >>> for item in outputs[0].recommendations:
        ...     print(f"Rank {item.rank}: item_{item.item_id} (score: {item.score:.4f})")
    """
    
    def __init__(
        self,
        model: str = "",
        embedding_path: Optional[str] = None,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        default_top_k: int = 10,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        # Model loading parameters
        device: Optional[str] = None,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        load_model: bool = True,
        # HSTU model architecture parameters (used when creating model from config without pretrained weights)
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        # Feature configuration (controls which fields are extracted from GRPrompt)
        uih_feature_names: Optional[List[str]] = None,
        candidate_feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize GRLLM.
        
        Note: Does not call super().__init__()
        
        Reason:
        1. LLM.__init__() creates a standard LLMEngine, but we need GRLLMEngine
        2. Fully overriding __init__() is consistent with vLLM design (LLM class has no parent)
        3. Maintains necessary attributes (llm_engine, request_counter, engine_class) initialization
        
        Reference: vllm/entrypoints/llm.py LLM.__init__()
        
        Args:
            model: Model name or path (HSTU model, empty string uses config to create)
            embedding_path: Embedding table path (optional)
            max_seq_len: Maximum sequence length
            max_candidates: Maximum number of candidates
            default_top_k: Default number of recommendations
            tensor_parallel_size: Tensor parallelism degree
            gpu_memory_utilization: GPU memory utilization
            device: Inference device (cuda/cpu), None for auto-selection
            embedding_dim: Embedding vector dimension
            hash_size: Embedding table hash size
            load_model: Whether to load and initialize the HSTU model
            num_layers: Number of HSTU layers (used without pretrained weights)
            num_heads: Number of attention heads (used without pretrained weights)
            hidden_dim: Hidden layer dimension (used without pretrained weights)
            attention_dim: Attention dimension (used without pretrained weights)
            **kwargs: Additional arguments
        """
        # 1. Build GR-specific engine arguments
        # Extract scheduler-related kwargs
        use_scheduler = kwargs.pop('use_scheduler', True)
        max_batch_size = kwargs.pop('max_batch_size', 32)
        scheduling_policy = kwargs.pop('scheduling_policy', 'fcfs')
        
        engine_args = GREngineArgs(
            model=model,
            embedding_path=embedding_path,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            default_top_k=default_top_k,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            embedding_dim=embedding_dim,
            hash_size=hash_size,
            load_model=load_model,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            # Scheduler configuration
            use_scheduler=use_scheduler,
            gr_max_batch_size=max_batch_size,
            scheduling_policy=scheduling_policy,
            # Feature configuration
            uih_feature_names=uih_feature_names or ["item_id", "timestamp", "rating", "action_type"],
            candidate_feature_names=candidate_feature_names or ["item_id", "category_id"],
        )
        
        # 2. Initialize engine (GRLLMEngine internally initializes processors)
        self._init_engine(engine_args)
        
        # 3. Save configuration (only for API layer validation)
        self.model = model
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.default_top_k = default_top_k
        
        # 4. Request counter (consistent with vLLM LLM)
        self.request_counter = Counter()
        
        # 5. Recommendation tasks don't need these vLLM attributes (keep as None for parent compatibility)
        self.default_sampling_params = None  # GR uses num_recommendations instead of SamplingParams
        self.supported_tasks = ("recommendation",)  # GR-specific task type
        self.io_processor = None  # GR doesn't use IO processor plugins
        
        logger.info(
            f"GRLLM initialized: model={model}, max_seq_len={max_seq_len}, "
            f"max_candidates={max_candidates}, default_top_k={default_top_k}"
        )
    
    def _init_engine(self, engine_args: GREngineArgs) -> None:
        """Initialize the engine."""
        self.llm_engine = GRLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )
        self.engine_class = type(self.llm_engine)
    
    # ==================== Recommendation Interface ====================
    
    def recommend(
        self,
        prompts: Union[GRPrompt, Sequence[GRPrompt]],
        *,
        use_tqdm: bool = True,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """
        Generate recommendation results.
        
        Args:
            prompts: Single or batch of GRPrompt
            use_tqdm: Whether to show progress bar
            top_k: Override default number of recommendations
        
        Returns:
            List of recommendation results, corresponding to input prompts order
        
        Example:
            >>> outputs = llm.recommend([
            ...     {"user_id": "u1", "user_history": [...], "candidates": [...]},
            ...     {"user_id": "u2", "user_history": [...], "candidates": [...]},
            ... ])
        """
        start_time = time.time()
        
        # Normalize to list
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        
        # Validate input
        for prompt in prompts:
            if not is_gr_prompt(prompt):
                raise ValueError(
                    f"Expected GRPrompt, got {type(prompt)}. "
                    "GRPrompt must contain 'user_id', 'user_history', and 'candidates'."
                )
        
        # Generate request IDs
        request_ids = [f"gr-{next(self.request_counter)}" for _ in prompts]
        
        # Add requests to engine (pass GRPrompt directly, processed by GRLLMEngine)
        for prompt, request_id in zip(prompts, request_ids):
            # Determine recommendation count: top_k parameter takes priority, then prompt value
            num_recs = top_k or prompt.get("num_recommendations")
            
            # Pass GRPrompt directly to GRLLMEngine
            self.llm_engine.add_gr_request(
                request_id=request_id,
                prompt=prompt,
                num_recommendations=num_recs,
            )
        
        # Run engine and collect results
        outputs = self._run_recommendation_engine(
            num_prompts=len(prompts),
            use_tqdm=use_tqdm,
        )
        
        elapsed = time.time() - start_time
        logger.debug(f"Recommendation completed in {elapsed:.3f}s for {len(prompts)} prompts")
        
        return outputs
    
    def _run_recommendation_engine(
        self,
        num_prompts: int,
        use_tqdm: bool,
    ) -> List[GRRequestOutput]:
        """Run engine and collect recommendation results."""
        outputs: List[GRRequestOutput] = []
        
        if use_tqdm:
            pbar = tqdm(total=num_prompts, desc="Generating recommendations")
        else:
            pbar = None
        
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            
            for output in step_outputs:
                if isinstance(output, GRRequestOutput) and output.finished:
                    outputs.append(output)
                    if pbar is not None:
                        pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        return outputs
    
    # ==================== Batch Interface ====================
    
    def recommend_batch(
        self,
        prompts: Sequence[GRPrompt],
        batch_size: int = 32,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """
        Batch recommendation (process large number of requests in batches).
        
        Suitable for offline batch inference scenarios.
        
        Args:
            prompts: List of GRPrompt
            batch_size: Number of requests per batch
            top_k: Number of recommendations
        
        Returns:
            List of recommendation results
        """
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = self.recommend(batch, top_k=top_k, use_tqdm=False)
            all_outputs.extend(outputs)
        
        return all_outputs
    
    # ==================== Engine Management ====================
    
    def get_engine(self) -> GRLLMEngine:
        """Get the underlying engine."""
        return self.llm_engine
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        if hasattr(self, 'llm_engine') and self.llm_engine is not None:
            self.llm_engine.shutdown()
        logger.info("GRLLM shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
