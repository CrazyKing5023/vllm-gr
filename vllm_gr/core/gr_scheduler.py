# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRScheduler: Scheduler for generative recommendation requests.

Implements a simplified scheduler for non-autoregressive GR models.
Unlike LLM inference, GR requests complete in a single step without
KV cache management.

Implements vLLM's SchedulerInterface for compatibility with EngineCore.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import time
import logging

from vllm.v1.core.sched.interface import SchedulerInterface

logger = logging.getLogger("vllm_gr.scheduler")


@dataclass
class GRSchedulerConfig:
    """GR Scheduler configuration."""
    
    max_batch_size: int = 32
    """Maximum number of requests per batch."""
    
    scheduling_policy: str = "fcfs"
    """Scheduling policy: 'fcfs' or 'priority'."""
    
    max_num_seqs: int = 256
    """Maximum number of sequences to hold in queue."""


@dataclass
class ScheduledGRRequest:
    """A request scheduled for execution."""
    
    request_id: str
    prompt_token_ids: List[int]  # user history item_ids
    candidate_item_ids: List[int]
    num_recommendations: int
    user_id: Any
    arrival_time: float
    priority: int = 0
    mm_inputs: Optional[Dict[str, Any]] = None


@dataclass
class GRSchedulerOutput:
    """Output from scheduler.schedule()."""
    
    scheduled_requests: List[ScheduledGRRequest]
    """Requests scheduled for this step."""
    
    num_scheduled_tokens: Dict[str, int]
    """Number of tokens (candidates) per request."""
    
    total_num_scheduled_tokens: int
    """Total tokens scheduled."""
    
    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


class GRScheduler(SchedulerInterface):
    """
    GR Scheduler - Implements vLLM SchedulerInterface.
    
    Key differences from vLLM Scheduler:
    - No KV cache management (GR doesn't use KV cache)
    - Single-step completion (non-autoregressive)
    - Simple batching based on candidate count
    - No preemption logic needed
    
    The scheduler manages:
    - Request queue (waiting requests)
    - Batch formation (respecting max_batch_size)
    - Priority-based scheduling (optional)
    """
    
    def __init__(
        self,
        config: Optional[GRSchedulerConfig] = None,
    ):
        """
        Initialize GRScheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or GRSchedulerConfig()
        
        # Request queues
        self._waiting_queue: List[ScheduledGRRequest] = []
        self._running: List[ScheduledGRRequest] = []
        self._finished_req_ids: Set[str] = set()
        self._aborted_req_ids: Set[str] = set()
        
        # Statistics
        self._num_requests_processed = 0
    
    def add_request(
        self,
        request: Any,  # Can be vLLM Request or our custom dict/params
        # Extended parameters for GR-specific add_request
        prompt_token_ids: Optional[List[int]] = None,
        candidate_item_ids: Optional[List[int]] = None,
        num_recommendations: Optional[int] = None,
        user_id: Any = None,
        priority: int = 0,
        mm_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a request to the waiting queue.
        
        Supports both:
        1. vLLM Request object (for SchedulerInterface compatibility)
        2. Extended GR parameters (for direct use)
        
        Args:
            request: vLLM Request object OR request_id string
            prompt_token_ids: User history item IDs (if request is string)
            candidate_item_ids: Candidate item IDs
            num_recommendations: Number of recommendations requested
            user_id: User identifier
            priority: Request priority (higher = more important)
            mm_inputs: Additional inputs for multi-modal support
        """
        # Handle vLLM Request object
        if hasattr(request, 'request_id'):
            # This is a vLLM Request object
            request_id = request.request_id
            prompt_token_ids = getattr(request, 'prompt_token_ids', [])
            # Try to get GR-specific data from request
            candidate_item_ids = getattr(request, 'candidate_item_ids', [])
            num_recommendations = getattr(request, 'num_recommendations', 10)
            user_id = getattr(request, 'user_id', None)
            priority = getattr(request, 'priority', 0)
            mm_inputs = getattr(request, 'mm_inputs', None)
        elif isinstance(request, str):
            # request is actually request_id string
            request_id = request
        else:
            # Assume it's a dict-like object
            request_id = str(request)
        
        if len(self._waiting_queue) >= self.config.max_num_seqs:
            logger.warning(
                f"Scheduler queue full ({self.config.max_num_seqs}), "
                f"dropping request {request_id}"
            )
            return
        
        scheduled_request = ScheduledGRRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids or [],
            candidate_item_ids=candidate_item_ids or [],
            num_recommendations=num_recommendations or 10,
            user_id=user_id,
            arrival_time=time.time(),
            priority=priority,
            mm_inputs=mm_inputs,
        )
        
        self._waiting_queue.append(scheduled_request)
        logger.debug(f"Added request {request_id} to queue")
    
    def add_gr_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        candidate_item_ids: List[int],
        num_recommendations: int,
        user_id: Any,
        priority: int = 0,
        mm_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a GR-specific request to the waiting queue.
        
        This is a convenience method with explicit GR parameters.
        
        Args:
            request_id: Unique request identifier
            prompt_token_ids: User history item IDs
            candidate_item_ids: Candidate item IDs
            num_recommendations: Number of recommendations requested
            user_id: User identifier
            priority: Request priority (higher = more important)
            mm_inputs: Additional inputs for multi-modal support
        """
        self.add_request(
            request=request_id,
            prompt_token_ids=prompt_token_ids,
            candidate_item_ids=candidate_item_ids,
            num_recommendations=num_recommendations,
            user_id=user_id,
            priority=priority,
            mm_inputs=mm_inputs,
        )
    
    def schedule(self) -> GRSchedulerOutput:
        """
        Schedule the next batch of requests.
        
        Returns:
            GRSchedulerOutput containing scheduled requests
        """
        if not self._waiting_queue:
            return GRSchedulerOutput(
                scheduled_requests=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
            )
        
        # Sort by priority if using priority scheduling
        if self.config.scheduling_policy == "priority":
            self._waiting_queue.sort(key=lambda r: -r.priority)
        
        # Select batch
        batch_size = min(self.config.max_batch_size, len(self._waiting_queue))
        batch = self._waiting_queue[:batch_size]
        self._waiting_queue = self._waiting_queue[batch_size:]
        
        # Move to running
        self._running = batch
        
        # Calculate token counts (using candidate count as proxy)
        num_scheduled_tokens = {}
        total_tokens = 0
        for req in batch:
            n_cand = len(req.candidate_item_ids)
            num_scheduled_tokens[req.request_id] = n_cand
            total_tokens += n_cand
        
        logger.debug(f"Scheduled {len(batch)} requests with {total_tokens} candidates")
        
        return GRSchedulerOutput(
            scheduled_requests=batch,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_tokens,
        )
    
    def update_from_output(
        self,
        scheduler_output: GRSchedulerOutput,
        model_outputs: List[Any],
    ) -> List[str]:
        """
        Update scheduler state after model execution.
        
        For GR, all requests complete in a single step.
        
        Args:
            scheduler_output: The scheduler output that was executed
            model_outputs: Model outputs for each request
            
        Returns:
            List of completed request IDs
        """
        completed_ids = []
        
        for req in scheduler_output.scheduled_requests:
            self._finished_req_ids.add(req.request_id)
            completed_ids.append(req.request_id)
            self._num_requests_processed += 1
        
        # Clear running queue
        self._running = []
        
        return completed_ids
    
    def abort_request(self, request_id: str) -> bool:
        """
        Abort a request.
        
        Args:
            request_id: Request to abort
            
        Returns:
            True if request was found and aborted
        """
        # Check waiting queue
        for i, req in enumerate(self._waiting_queue):
            if req.request_id == request_id:
                self._waiting_queue.pop(i)
                self._aborted_req_ids.add(request_id)
                return True
        
        # Check running queue
        for i, req in enumerate(self._running):
            if req.request_id == request_id:
                self._running.pop(i)
                self._aborted_req_ids.add(request_id)
                return True
        
        return False
    
    def finish_requests(
        self,
        request_ids: Any,  # Union[str, Iterable[str]]
        finished_status: Any = None,  # RequestStatus (optional for GR)
    ) -> None:
        """
        Mark requests as finished.
        
        Args:
            request_ids: Single request ID or iterable of request IDs
            finished_status: The finished status (unused for GR)
        """
        # Handle single string or iterable
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        
        request_id_set = set(request_ids)
        self._waiting_queue = [
            r for r in self._waiting_queue 
            if r.request_id not in request_id_set
        ]
        self._running = [
            r for r in self._running 
            if r.request_id not in request_id_set
        ]
        self._finished_req_ids.update(request_ids)
    
    def has_requests(self) -> bool:
        """Check if there are pending or running requests."""
        return len(self._waiting_queue) > 0 or len(self._running) > 0
    
    def has_unfinished_requests(self) -> bool:
        """Alias for has_requests()."""
        return self.has_requests()
    
    def get_num_unfinished_requests(self) -> int:
        """Get total number of unfinished requests."""
        return len(self._waiting_queue) + len(self._running)
    
    def get_num_waiting_requests(self) -> int:
        """Get number of waiting requests."""
        return len(self._waiting_queue)
    
    def get_num_running_requests(self) -> int:
        """Get number of running requests."""
        return len(self._running)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "num_waiting_requests": len(self._waiting_queue),
            "num_running_requests": len(self._running),
            "num_requests_processed": self._num_requests_processed,
            "num_aborted_requests": len(self._aborted_req_ids),
            "max_batch_size": self.config.max_batch_size,
            "scheduling_policy": self.config.scheduling_policy,
        }
    
    # =========================================================================
    # SchedulerInterface required abstract methods
    # =========================================================================
    
    def has_finished_requests(self) -> bool:
        """
        Returns True if there are finished requests that need to be cleared.
        
        For GR, requests are completed in a single step, so we track
        finished requests that haven't been cleared yet.
        """
        return len(self._finished_req_ids) > 0
    
    def get_request_counts(self) -> Tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return (len(self._running), len(self._waiting_queue))
    
    def update_draft_token_ids(self, draft_token_ids: Any) -> None:
        """
        Update draft token IDs for speculative decoding.
        
        Not applicable for GR models (non-autoregressive).
        """
        # GR models don't use speculative decoding
        pass
    
    def reset_prefix_cache(self) -> bool:
        """
        Reset the prefix cache for KV cache.
        
        Not applicable for GR models (no KV cache).
        """
        # GR models don't use KV cache
        return True
    
    def make_stats(self) -> Optional[Any]:
        """
        Make a SchedulerStats object for logging.
        
        Returns None for GR scheduler (basic stats available via get_stats()).
        """
        # Return None - GR scheduler doesn't produce standard vLLM stats
        return None
    
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self._waiting_queue.clear()
        self._running.clear()
        self._finished_req_ids.clear()
        self._aborted_req_ids.clear()
        logger.info("GRScheduler shutdown complete")
