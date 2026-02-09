# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Output data types for generative recommendation.

This module defines the output types for the GRLLM API:
- RecommendedItem: Single recommended item with score and rank
- GRRequestOutput: Complete recommendation response
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union, List, Dict

# Required imports - these are mandatory dependencies
import torch


@dataclass
class RecommendedItem:
    """
    Single recommended item.
    
    Attributes:
        item_id: Item ID
        score: Prediction score
        rank: Ranking (starting from 1)
        features: Item features (optional)
    """
    
    item_id: Union[str, int]
    """Item ID"""
    
    score: float
    """Prediction score"""
    
    rank: int
    """Ranking (starting from 1)"""
    
    features: Optional[Dict[str, Any]] = None
    """Item features (optional)"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "rank": self.rank,
            "features": self.features,
        }


@dataclass
class GRRequestOutput:
    """
    Recommendation request output.
    
    Design principle: Similar to vLLM's RequestOutput,
    but contains recommendation result list instead of generated tokens.
    
    Attributes:
        request_id: Unique request identifier
        user_id: User ID
        recommendations: Recommendation result list (sorted by score descending)
        finished: Whether completed
        all_scores: Prediction scores for all candidates (optional)
        user_embedding: User representation vector (optional)
        metrics: Inference metrics
    """
    
    request_id: str
    """Unique request identifier"""
    
    user_id: Union[str, int]
    """User ID"""
    
    recommendations: List[RecommendedItem]
    """Recommendation result list (sorted by score descending)"""
    
    finished: bool = True
    """Whether completed (recommendation models typically complete in one step)"""
    
    # Raw prediction data (optional, for debugging or further processing)
    all_scores: Optional[torch.Tensor] = None
    """Prediction scores for all candidates [num_candidates]"""
    
    user_embedding: Optional[torch.Tensor] = None
    """User representation vector [embed_dim]"""
    
    # Inference metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    """Inference metrics (latency, batch_size, etc.)"""
    
    def __repr__(self) -> str:
        return (
            f"GRRequestOutput(request_id={self.request_id!r}, "
            f"user_id={self.user_id!r}, "
            f"num_recommendations={len(self.recommendations)}, "
            f"finished={self.finished})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for JSON serialization)."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "finished": self.finished,
            "metrics": self.metrics,
        }
    
    @property
    def top_recommendation(self) -> Optional[RecommendedItem]:
        """Get the top-ranked recommendation."""
        return self.recommendations[0] if self.recommendations else None
    
    @property
    def item_ids(self) -> List[Union[str, int]]:
        """Get the list of all recommended item IDs."""
        return [item.item_id for item in self.recommendations]
    
    @property
    def scores(self) -> List[float]:
        """Get the list of all recommended item scores."""
        return [item.score for item in self.recommendations]
