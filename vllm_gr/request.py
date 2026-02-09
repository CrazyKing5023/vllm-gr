# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRRequest: Extended request class for generative recommendation.

This module defines GRRequest which extends vLLM's Request class
with recommendation-specific fields.
"""

from typing import Any, Optional, Union, List, Dict
from dataclasses import dataclass, field


@dataclass
class GRRequest:
    """
    Generative recommendation request.
    
    Design principles:
    1. Contains recommendation-specific fields (gr_inputs, original_candidates, etc.)
    2. When vLLM is available, replaces vLLM's Request class via Monkey Patch
    3. When vLLM is unavailable, used as a standalone request class
    
    Key fields:
    - gr_inputs: Processed input (KJT format)
    - original_candidates: Original candidate list (for output mapping)
    
    Attributes:
        request_id: Unique request identifier
        gr_inputs: Processed GRInputs
        user_id: User ID
        num_recommendations: Number of recommendations
        original_candidates: Original candidate list
        arrival_time: Request arrival time
        priority: Request priority
    """
    
    request_id: str
    """Unique request identifier"""
    
    gr_inputs: Optional[Any] = None  # GRInputs type, use Any to avoid circular import
    """Processed recommendation model input"""
    
    user_id: Optional[Union[str, int]] = None
    """User ID"""
    
    num_recommendations: int = 10
    """Number of recommendations requested"""
    
    original_candidates: List[Dict[str, Any]] = field(default_factory=list)
    """Original candidate list (for output mapping)"""
    
    arrival_time: Optional[float] = None
    """Request arrival time (Unix timestamp)"""
    
    priority: int = 0
    """Request priority (higher value = higher priority)"""
    
    # vLLM Request compatibility fields
    prompt: Optional[str] = None
    """Original prompt (compatibility field, not used for recommendation tasks)"""
    
    prompt_token_ids: Optional[List[int]] = None
    """Token IDs (compatibility field, not used for recommendation tasks)"""
    
    @property
    def is_gr_request(self) -> bool:
        """Check if this is a recommendation request."""
        return self.gr_inputs is not None
    
    @property
    def num_candidates(self) -> int:
        """Get the number of candidates."""
        return len(self.original_candidates)
    
    def __repr__(self) -> str:
        return (
            f"GRRequest(request_id={self.request_id!r}, "
            f"user_id={self.user_id!r}, "
            f"num_recommendations={self.num_recommendations}, "
            f"num_candidates={self.num_candidates})"
        )


# Try to inherit from vLLM's Request class (if available)
_VLLM_REQUEST_AVAILABLE = False

try:
    from vllm.v1.request import Request as VLLMRequest
    _VLLM_REQUEST_AVAILABLE = True
    
    class GRRequestWithVLLM(VLLMRequest):
        """
        GRRequest that inherits from vLLM Request.
        
        Used when vLLM is available for full vLLM compatibility.
        """
        
        def __init__(
            self,
            request_id: str,
            # GR-specific fields
            gr_inputs: Optional[Any] = None,
            user_id: Optional[Union[str, int]] = None,
            num_recommendations: int = 10,
            original_candidates: Optional[List[Dict]] = None,
            arrival_time: Optional[float] = None,
            priority: int = 0,
            # vLLM Request fields
            **kwargs,
        ):
            # Call parent constructor (if needed)
            # Note: vLLM v1 Request signature may differ, needs adaptation
            try:
                super().__init__(request_id=request_id, **kwargs)
            except TypeError:
                # If parent signature doesn't match, only set request_id
                self.request_id = request_id
            
            # GR-specific fields
            self.gr_inputs = gr_inputs
            self.user_id = user_id
            self.num_recommendations = num_recommendations
            self.original_candidates = original_candidates or []
            self.arrival_time = arrival_time
            self.priority = priority
        
        @property
        def is_gr_request(self) -> bool:
            """Check if this is a recommendation request."""
            return self.gr_inputs is not None
        
        @property
        def num_candidates(self) -> int:
            """Get the number of candidates."""
            return len(self.original_candidates)
    
    # Export inherited version
    GRRequest = GRRequestWithVLLM  # type: ignore

except ImportError:
    # vLLM unavailable, use standalone version
    pass


def is_gr_request(request: Any) -> bool:
    """
    Check if the request is a GRRequest.
    
    Args:
        request: Request object to check
        
    Returns:
        bool: Whether it is a GRRequest
    """
    return hasattr(request, "is_gr_request") and request.is_gr_request
