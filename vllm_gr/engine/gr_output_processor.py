# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GROutputProcessor: Output processor for generative recommendation.

Converts vLLM's PoolingRequestOutput (from EngineCore) into
GRRequestOutput with Top-K selection and score ranking.

This processor tracks request metadata (original candidates, user_id,
num_recommendations) and maps model prediction scores back to candidate
items to produce the final recommendation list.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict
import logging

import torch

from vllm_gr.outputs import GRRequestOutput, RecommendedItem

logger = logging.getLogger("vllm_gr.output_processor")


@dataclass
class GRProcessedOutputs:
    """Processed outputs from GR engine."""
    request_outputs: List[GRRequestOutput]
    iteration_stats: Optional[Any] = None


class GROutputProcessor:
    """
    GR Output Processor — converts engine outputs to GRRequestOutput.

    Responsibilities:
    1. Track active requests with their metadata (original_candidates,
       user_id, num_recommendations).
    2. Convert model prediction scores to Top-K recommendations.
    3. Map prediction indices back to original candidate items.

    Lifecycle:
    - add_request(): called by GRLLMEngine when a new request is added
    - process_outputs(): called by GRLLMEngine.step() to convert outputs
    - Automatically removes completed requests
    """

    def __init__(
        self,
        default_top_k: int = 10,
        log_stats: bool = False,
    ):
        self.default_top_k = default_top_k
        self.log_stats = log_stats
        # Track active requests: request_id -> metadata dict
        self._active_requests: Dict[str, Dict[str, Any]] = {}

    def add_request(
        self,
        request_id: str,
        user_id: Any,
        num_recommendations: int,
        original_candidates: List[Dict],
    ) -> None:
        """
        Register a new request for output tracking.

        Args:
            request_id: Unique request ID
            user_id: User identifier
            num_recommendations: Number of Top-K results requested
            original_candidates: Original candidate dicts for output mapping
        """
        self._active_requests[request_id] = {
            "user_id": user_id,
            "num_recommendations": num_recommendations,
            "original_candidates": original_candidates,
        }

    def remove_request(self, request_id: str) -> None:
        """Remove a request from tracking."""
        self._active_requests.pop(request_id, None)

    def has_request(self, request_id: str) -> bool:
        """Check if a request is being tracked."""
        return request_id in self._active_requests

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self._active_requests) > 0

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return len(self._active_requests)

    def process_outputs(
        self,
        engine_outputs: List[Any],
    ) -> GRProcessedOutputs:
        """
        Process EngineCore outputs into GRRequestOutputs.

        Handles both PoolingRequestOutput (normal path) and
        RequestOutput (fallback path).

        Args:
            engine_outputs: Outputs from engine_core / super().step()

        Returns:
            GRProcessedOutputs with recommendation results
        """
        request_outputs: List[GRRequestOutput] = []

        for output in engine_outputs:
            request_id = getattr(output, "request_id", None)
            if request_id is None or request_id not in self._active_requests:
                continue

            metadata = self._active_requests[request_id]
            user_id = metadata["user_id"]
            num_recs = metadata["num_recommendations"]
            candidates = metadata["original_candidates"]

            # Extract prediction scores from the engine output
            scores = self._extract_scores(output, len(candidates))

            # Create GRRequestOutput with Top-K selection
            gr_output = self._create_gr_output(
                request_id=request_id,
                user_id=user_id,
                scores=scores,
                candidates=candidates,
                num_recommendations=num_recs,
            )
            request_outputs.append(gr_output)

            # Remove completed request
            if gr_output.finished:
                del self._active_requests[request_id]

        return GRProcessedOutputs(
            request_outputs=request_outputs,
        )

    def _extract_scores(
        self,
        output: Any,
        num_candidates: int,
    ) -> Union[torch.Tensor, List[float]]:
        """
        Extract prediction scores from an engine output.

        Tries multiple paths:
        1. PoolingRequestOutput.outputs.data (tensor)
        2. RequestOutput.outputs[0].data
        3. Fallback: descending placeholder scores
        """
        # PoolingRequestOutput path
        if hasattr(output, "outputs") and output.outputs is not None:
            out = output.outputs
            # PoolingOutput has .data attribute
            if hasattr(out, "data") and out.data is not None:
                return out.data
            # List of outputs (RequestOutput)
            if isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                if hasattr(first, "data") and first.data is not None:
                    return first.data

        # Fallback: generate descending placeholder scores
        logger.warning(
            f"Could not extract scores from output type "
            f"{type(output).__name__}, using placeholder scores"
        )
        return list(range(num_candidates, 0, -1))

    def _create_gr_output(
        self,
        request_id: str,
        user_id: Any,
        scores: Union[torch.Tensor, List[float]],
        candidates: List[Dict],
        num_recommendations: int,
    ) -> GRRequestOutput:
        """
        Create GRRequestOutput from prediction scores and candidates.

        Performs Top-K selection via torch.topk or Python sorting.
        """
        if isinstance(scores, torch.Tensor):
            scores_float = scores.float()
            top_k = min(
                num_recommendations, len(candidates), len(scores_float)
            )
            if top_k > 0:
                top_scores, top_indices = torch.topk(scores_float, k=top_k)
                top_scores_list = top_scores.tolist()
                top_indices_list = top_indices.tolist()
            else:
                top_scores_list = []
                top_indices_list = []
            all_scores = scores
        elif isinstance(scores, (list, tuple)):
            indexed = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )
            top_k = min(num_recommendations, len(candidates), len(scores))
            top_indices_list = [idx for idx, _ in indexed[:top_k]]
            top_scores_list = [s for _, s in indexed[:top_k]]
            all_scores = None
        else:
            return GRRequestOutput(
                request_id=request_id,
                user_id=user_id,
                recommendations=[],
                finished=True,
            )

        # Build recommendation list
        recommendations = []
        for rank, (idx, score) in enumerate(
            zip(top_indices_list, top_scores_list)
        ):
            if idx < len(candidates):
                candidate = candidates[idx]
                recommendations.append(
                    RecommendedItem(
                        item_id=candidate.get("item_id", idx),
                        score=float(score),
                        rank=rank + 1,
                        features=candidate,
                    )
                )

        return GRRequestOutput(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            all_scores=all_scores,
            finished=True,
        )
