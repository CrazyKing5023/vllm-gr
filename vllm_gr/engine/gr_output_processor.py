# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GROutputProcessor: Output processor for generative recommendation.

Converts model predictions to GRRequestOutput with Top-K sampling.
Inherits from vLLM's OutputProcessor to maintain API compatibility.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict
import logging

import torch

from vllm.v1.engine.output_processor import OutputProcessor

from vllm_gr.outputs import GRRequestOutput, RecommendedItem
from vllm_gr.request import GRRequest

logger = logging.getLogger("vllm_gr.output_processor")


@dataclass
class GRProcessedOutputs:
    """Processed outputs from GR engine."""
    request_outputs: List[GRRequestOutput]
    iteration_stats: Optional[Any] = None


class GROutputProcessor(OutputProcessor):
    """
    GR Output Processor - Inherits from vLLM OutputProcessor.
    
    Responsibilities:
    1. Track active requests (save original_candidates for output mapping)
    2. Convert model prediction scores to GRRequestOutput
    3. Top-K sampling (sort by score, select best recommendations)
    
    Notes:
    - Created and managed by GRLLMEngine
    - GRLLM does not directly create this processor
    """
    
    def __init__(
        self,
        default_top_k: int = 10,
        log_stats: bool = False,
    ):
        """
        Args:
            default_top_k: Default number of recommendations
            log_stats: Whether to log statistics
        """
        self.default_top_k = default_top_k
        self.log_stats = log_stats
        
        # Track active requests
        self._active_requests: Dict[str, Dict] = {}
    
    def add_gr_request(
        self,
        gr_request: GRRequest,
        original_candidates: List[Dict],
    ) -> None:
        """
        Add request to tracking list.
        
        Called by GRLLMEngine.add_gr_request().
        
        Args:
            gr_request: GRRequest object
            original_candidates: Original candidate list (for output mapping)
        """
        self._active_requests[gr_request.request_id] = {
            "request": gr_request,
            "original_candidates": original_candidates,
        }
    
    def remove_request(self, request_id: str) -> None:
        """Remove a request."""
        if request_id in self._active_requests:
            del self._active_requests[request_id]
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self._active_requests) > 0
    
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return len(self._active_requests)
    
    def get_request_info(self, request_id: str) -> Optional[Dict]:
        """Get request information."""
        return self._active_requests.get(request_id)
    
    def process_gr_outputs(
        self,
        engine_outputs: List[Any],
        engine_core_timestamp: Optional[float] = None,
    ) -> GRProcessedOutputs:
        """
        Process EngineCore output to GRRequestOutput.
        
        Called by GRLLMEngine.step().
        
        Args:
            engine_outputs: Output list returned by EngineCore
            engine_core_timestamp: Engine timestamp
            
        Returns:
            GRProcessedOutputs: Processed outputs
        """
        request_outputs = []
        
        for output in engine_outputs:
            request_id = getattr(output, 'request_id', None)
            if request_id is None or request_id not in self._active_requests:
                continue
            
            req_info = self._active_requests[request_id]
            original_candidates = req_info["original_candidates"]
            gr_request = req_info["request"]
            
            # Get prediction scores
            predictions = getattr(output, 'predictions', None)
            if predictions is None:
                predictions = getattr(output, 'scores', None)
            
            if predictions is None:
                # Skip if no predictions
                continue
            
            # Extract scores from model output and generate recommendations
            gr_output = self._create_gr_output(
                request_id=request_id,
                user_id=gr_request.user_id,
                predictions=predictions,
                original_candidates=original_candidates,
                num_recommendations=gr_request.num_recommendations,
            )
            
            request_outputs.append(gr_output)
            
            # Remove after completion
            if gr_output.finished:
                del self._active_requests[request_id]
        
        return GRProcessedOutputs(
            request_outputs=request_outputs,
            iteration_stats=None,
        )
    
    def create_output_from_scores(
        self,
        request_id: str,
        scores: Any,
    ) -> Optional[GRRequestOutput]:
        """
        Create output directly from scores (for simplified calls).
        
        Args:
            request_id: Request ID
            scores: Prediction scores (tensor or list)
            
        Returns:
            GRRequestOutput or None
        """
        if request_id not in self._active_requests:
            return None
        
        req_info = self._active_requests[request_id]
        original_candidates = req_info["original_candidates"]
        gr_request = req_info["request"]
        
        gr_output = self._create_gr_output(
            request_id=request_id,
            user_id=gr_request.user_id,
            predictions=scores,
            original_candidates=original_candidates,
            num_recommendations=gr_request.num_recommendations,
        )
        
        # Remove after completion
        if gr_output.finished:
            del self._active_requests[request_id]
        
        return gr_output
    
    def _create_gr_output(
        self,
        request_id: str,
        user_id: Any,
        predictions: Any,
        original_candidates: List[Dict],
        num_recommendations: int,
    ) -> GRRequestOutput:
        """
        Create GRRequestOutput from model predictions.
        
        Args:
            request_id: Request ID
            user_id: User ID
            predictions: Prediction scores (tensor or list)
            original_candidates: Original candidate list
            num_recommendations: Number of recommendations
            
        Returns:
            GRRequestOutput: Recommendation results
        """
        # Convert predictions to processable format
        if isinstance(predictions, torch.Tensor):
            scores = predictions.float()
            top_k = min(num_recommendations, len(original_candidates), len(scores))
            top_scores, top_indices = torch.topk(scores, k=top_k)
            top_scores_list = top_scores.tolist()
            top_indices_list = top_indices.tolist()
            all_scores = predictions
        elif isinstance(predictions, (list, tuple)):
            # Use Python sorting
            scores = list(predictions)
            indexed_scores = [(i, s) for i, s in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_k = min(num_recommendations, len(original_candidates), len(scores))
            top_indices_list = [idx for idx, _ in indexed_scores[:top_k]]
            top_scores_list = [s for _, s in indexed_scores[:top_k]]
            all_scores = None
        else:
            # Unprocessable format, return empty result
            return GRRequestOutput(
                request_id=request_id,
                user_id=user_id,
                recommendations=[],
                finished=True,
            )
        
        # Build recommendation list
        recommendations = []
        for rank, (idx, score) in enumerate(zip(top_indices_list, top_scores_list)):
            if idx < len(original_candidates):
                original_item = original_candidates[idx]
                item_id = original_item.get("item_id", idx)
                
                recommendations.append(RecommendedItem(
                    item_id=item_id,
                    score=float(score),
                    rank=rank + 1,
                    features=original_item.get("features"),
                ))
        
        return GRRequestOutput(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            all_scores=all_scores,
            finished=True,
        )
