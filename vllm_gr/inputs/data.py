# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Core data types for generative recommendation.

This module defines the input/output types for the GRLLM API:
- GRPrompt: User-friendly API input format
- GRInputs: Processed internal format (with KeyedJaggedTensor)
"""

from typing import Any, Literal, Optional, Union, List, Dict, TYPE_CHECKING

# Python 3.11+ has TypedDict and NotRequired in typing
# For older versions, fall back to typing_extensions
try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired

# Required imports - these are mandatory dependencies
import torch


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


# TorchRec import - optional for KeyedJaggedTensor
try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    _TORCHREC_AVAILABLE = True
except ImportError:
    _TORCHREC_AVAILABLE = False
    KeyedJaggedTensor = Any  # type: ignore


class GRInputs(TypedDict):
    """
    Processed recommendation model input (unified format).
    
    Design principles:
    1. Uses KeyedJaggedTensor format, consistent with generative-recommenders internal format
    2. Similar to vLLM's TokenInputs/EmbedsInputs, but for recommendation scenarios
    
    Note: Requires torchrec library for KeyedJaggedTensor, requires torch for Tensor
    """
    
    type: Literal["gr_inputs"]
    """Input type identifier"""
    
    # ===== Core Tensors (KJT format) =====
    uih_features_kjt: Any  # KeyedJaggedTensor
    """User history features (KeyedJaggedTensor)"""
    
    candidates_features_kjt: Any  # KeyedJaggedTensor
    """Candidate item features (KeyedJaggedTensor)"""
    
    # ===== Length Information =====
    sequence_lengths: torch.Tensor
    """Valid sequence lengths [batch_size], int32"""
    
    candidate_counts: torch.Tensor
    """Candidate counts [batch_size], int32"""
    
    # ===== Metadata =====
    user_id: Union[str, int]
    """Original user ID"""
    
    num_recommendations: int
    """Number of recommendations requested"""
    
    context: NotRequired[Dict[str, Any]]
    """Context features (optional)"""


def gr_inputs(
    uih_features_kjt: Any,  # KeyedJaggedTensor
    candidates_features_kjt: Any,  # KeyedJaggedTensor
    sequence_lengths: torch.Tensor,
    candidate_counts: torch.Tensor,
    user_id: Union[str, int],
    num_recommendations: int = 10,
    context: Optional[Dict[str, Any]] = None,
) -> GRInputs:
    """
    Factory function to construct GRInputs.
    
    Args:
        uih_features_kjt: User history features KJT
        candidates_features_kjt: Candidate item features KJT
        sequence_lengths: Sequence length tensor
        candidate_counts: Candidate count tensor
        user_id: User ID
        num_recommendations: Number of recommendations
        context: Context features (optional)
        
    Returns:
        GRInputs: Constructed input object
    """
    inputs: GRInputs = {
        "type": "gr_inputs",
        "uih_features_kjt": uih_features_kjt,
        "candidates_features_kjt": candidates_features_kjt,
        "sequence_lengths": sequence_lengths,
        "candidate_counts": candidate_counts,
        "user_id": user_id,
        "num_recommendations": num_recommendations,
    }
    
    if context is not None:
        inputs["context"] = context
    
    return inputs
