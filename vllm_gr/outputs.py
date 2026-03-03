# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Output data types for generative recommendation.

- RecommendedItem: a single ranked candidate with score
- GRRequestOutput: full recommendation response, extends PoolingRequestOutput
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.outputs import PoolingOutput, PoolingRequestOutput


@dataclass
class RecommendedItem:
    """Single recommended item with score and rank."""

    item_id: str | int
    score: float
    rank: int
    features: dict[str, Any] | None = None
    task_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "item_id": self.item_id,
            "score": self.score,
            "rank": self.rank,
            "features": self.features,
        }
        if self.task_scores is not None:
            result["task_scores"] = self.task_scores
        return result


class GRRequestOutput(PoolingRequestOutput):
    """Recommendation output extending PoolingRequestOutput.

    Adds recommendation-specific fields (user_id, ranked recommendations,
    raw scores) while remaining compatible with vLLM's output pipeline.
    """

    def __init__(
        self,
        request_id: str,
        user_id: Any = None,
        recommendations: list[RecommendedItem] | None = None,
        finished: bool = True,
        all_scores: torch.Tensor | None = None,
        prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int = 0,
    ):
        outputs = PoolingOutput(data=all_scores if all_scores is not None
                                else torch.tensor([]))
        super().__init__(
            request_id=request_id,
            outputs=outputs,
            prompt_token_ids=prompt_token_ids or [],
            num_cached_tokens=num_cached_tokens,
            finished=finished,
        )

        self.user_id = user_id
        self.recommendations = recommendations or []
        self.all_scores = all_scores

    def __repr__(self) -> str:
        return (
            f"GRRequestOutput(request_id={self.request_id!r}, "
            f"user_id={self.user_id!r}, "
            f"num_recommendations={len(self.recommendations)}, "
            f"finished={self.finished})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "finished": self.finished,
        }

    @property
    def top_recommendation(self) -> RecommendedItem | None:
        return self.recommendations[0] if self.recommendations else None

    @property
    def item_ids(self) -> list[str | int]:
        return [item.item_id for item in self.recommendations]

    @property
    def scores(self) -> list[float]:
        return [item.score for item in self.recommendations]
