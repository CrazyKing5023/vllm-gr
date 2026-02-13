# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Core data types for generative recommendation.

This module defines the input types for the GRLLM API:
- GRPrompt: User-friendly API input format
- ItemInteraction: User-item interaction record
- CandidateItem: Candidate item for ranking
"""

from typing import Any, Union, List, Dict

# Python 3.11+ has TypedDict and NotRequired in typing
# For older versions, fall back to typing_extensions
try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired

class ItemInteraction(TypedDict):
    """
    User-item interaction record.
    
    Corresponds to a row of uih_features_kjt in generative-recommenders.
    """
    
    item_id: int
    """Item ID"""
    
    timestamp: NotRequired[int]
    """Interaction timestamp (Unix timestamp or relative time)"""
    
    rating: NotRequired[int]
    """Rating (e.g., 1-5)"""
    
    action_type: NotRequired[int]
    """Action type index (0=view, 1=click, 2=purchase, ...)"""
    
    category_id: NotRequired[int]
    """Item category ID"""


class CandidateItem(TypedDict):
    """
    Candidate item.
    
    Corresponds to a row of candidates_features_kjt in generative-recommenders.
    """
    
    item_id: int
    """Item ID"""
    
    category_id: NotRequired[int]
    """Item category ID"""


class GRPrompt(TypedDict):
    """
    Prompt format for generative recommendation - user-friendly API layer format.
    
    Design principles:
    1. Similar to vLLM's TokensPrompt/TextPrompt
    2. Internally converted to TorchRec KeyedJaggedTensor
    
    Example:
        >>> prompt: GRPrompt = {
        ...     "user_id": 123,
        ...     "user_history": [
        ...         {"item_id": 1, "timestamp": 1000, "rating": 4},
        ...         {"item_id": 50, "timestamp": 2000, "rating": 5},
        ...     ],
        ...     "candidates": [{"item_id": i} for i in range(1, 101)],
        ...     "num_recommendations": 10,
        ... }
    """
    
    user_id: Union[str, int]
    """User unique identifier"""
    
    user_history: List[ItemInteraction]
    """User history interaction sequence (sorted by time, oldest to newest)"""
    
    candidates: List[CandidateItem]
    """Candidate item list"""
    
    num_recommendations: NotRequired[int]
    """Number of recommendations requested, default 10"""
    
    context: NotRequired[Dict[str, Any]]
    """Context features (optional)"""


def is_gr_prompt(prompt: Any) -> bool:
    """
    Check if the object is a GRPrompt type.
    
    Args:
        prompt: Object to check
        
    Returns:
        bool: Whether it is a valid GRPrompt
    """
    return (
        isinstance(prompt, dict) 
        and "user_id" in prompt 
        and "user_history" in prompt
        and "candidates" in prompt
    )
