# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Output data types for generative recommendation.

- RecommendedItem: a single candidate with score metadata
- GRRequestOutput: full recommendation response, extends PoolingRequestOutput
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.outputs import PoolingOutput, PoolingRequestOutput


@dataclass
class RecommendedItem:
    """Single candidate item with score metadata."""

    item_id: str | int
    score: float
    features: dict[str, Any] | None = None
    task_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "item_id": self.item_id,
            "score": self.score,
            "features": self.features,
        }
        if self.task_scores is not None:
            result["task_scores"] = self.task_scores
        return result


class GRRequestOutput(PoolingRequestOutput):
    """Recommendation output extending PoolingRequestOutput.

    Adds recommendation-specific fields (user_id, candidate scores, raw
    scores) while remaining compatible with vLLM's output pipeline.
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
            f"num_candidates={len(self.recommendations)}, "
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
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda item: item.score)

    @property
    def item_ids(self) -> list[str | int]:
        return [item.item_id for item in self.recommendations]

    @property
    def scores(self) -> list[float]:
        return [item.score for item in self.recommendations]


# ------------------------------------------------------------------
# Phase 2: Adapter-level output type hierarchy
# ------------------------------------------------------------------

class GRModelOutput:
    """Base class for adapter-level model outputs.

    Adapters produce GRModelOutput instances from ``process_output()``.
    These are intermediate representations — the engine layer wraps them
    into GRRequestOutput for API-level consumption.

    IMPORTANT: GRRequestOutput (above) is the final API type and is NOT
    a subclass of GRModelOutput.  The hierarchy is:

        GRModelOutput (adapter internal)
          +-- ScoringOutput  (candidate scoring)
          +-- EmbeddingOutput (item embeddings)
          +-- TextOutput      (generative text)

        GRRequestOutput (API level, extends PoolingRequestOutput)
          +-- Used by gr.py, async_gr_engine.py, serving.py

    Design doc's ``request_id`` / ``user_id`` / ``output_type`` /
    ``finished`` fields on GRModelOutput are deferred to a later
    "output contract unification" stage.
    """


@dataclass
class ScoringOutput(GRModelOutput):
    """Output from a scoring model — ranked candidate items.

    Produced by adapters that implement ``GRScoringModel`` (e.g. HSTU).
    ``recommendations`` uses the same ``list[RecommendedItem]`` type as
    ``GRRequestOutput.recommendations`` to facilitate future unification.
    """

    recommendations: list[RecommendedItem] = field(default_factory=list)
    all_scores: torch.Tensor | None = None
    request_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingOutput(GRModelOutput):
    """Output from an embedding model — item embeddings and similarities.

    Reserved for future use (Phase 4+).  Produced by adapters that
    implement ``GREmbeddingModel`` (e.g. SASRec embedding mode).
    """

    embeddings: torch.Tensor | None = None
    item_ids: list[str | int] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    request_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextOutput(GRModelOutput):
    """Output from a generative model — decoded text.

    Reserved for future use (Phase 5+).  Produced by adapters that
    support generative inference (e.g. P5).
    """

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    request_meta: dict[str, Any] = field(default_factory=dict)
