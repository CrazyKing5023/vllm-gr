# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRProcessor: Input processor for generative recommendation.

Converts GRPrompt to GRRequest with KeyedJaggedTensor format inputs.
"""

from typing import Any, Optional, Union, List, Dict
import time

# Required imports - these are mandatory dependencies
import torch

# TorchRec import - optional for KeyedJaggedTensor
try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    _TORCHREC_AVAILABLE = True
except ImportError:
    _TORCHREC_AVAILABLE = False
    KeyedJaggedTensor = None  # type: ignore

from vllm_gr.inputs.data import GRPrompt, GRInputs
from vllm_gr.request import GRRequest


class GRProcessor:
    """
    GR Unified Input Processor.
    
    Responsibilities:
    1. Convert GRPrompt to GRInputs (containing KJT)
    2. Create GRRequest
    
    Data format correspondence with generative-recommenders:
    - GRPrompt.user_history -> uih_features_kjt
    - GRPrompt.candidates -> candidates_features_kjt
    
    Notes:
    - Created and managed by GRLLMEngine
    - GRLLM does not directly create this processor
    - torch is a required dependency
    - TorchRec is optional; uses simplified format when unavailable
    """
    
    def __init__(
        self,
        vllm_config: Optional[Any] = None,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        uih_feature_names: Optional[List[str]] = None,
        candidate_feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            vllm_config: vLLM configuration (optional)
            max_seq_len: Maximum sequence length
            max_candidates: Maximum number of candidates
            uih_feature_names: User history feature name list
            candidate_feature_names: Candidate item feature name list
        """
        self.vllm_config = vllm_config
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        
        # Feature name configuration (corresponds to generative-recommenders)
        self.uih_feature_names = uih_feature_names or ["item_id"]
        self.candidate_feature_names = candidate_feature_names or ["candidate_item_id"]
    
    def process_gr_inputs(
        self,
        request_id: str,
        prompt: GRPrompt,
        num_recommendations: int,
        arrival_time: Optional[float] = None,
        priority: int = 0,
    ) -> GRRequest:
        """
        Process GRPrompt to generate GRRequest.
        
        This is the main entry method, called by GRLLMEngine.add_gr_request().
        
        Args:
            request_id: Request ID
            prompt: User request in GRPrompt format
            num_recommendations: Number of recommendations
            arrival_time: Request arrival time
            priority: Request priority
        
        Returns:
            GRRequest: Request object containing processed inputs
        """
        if arrival_time is None:
            arrival_time = time.time()
        
        # 1. Encode user history
        user_history = prompt.get("user_history", [])
        uih_kjt = self._encode_user_history(user_history)
        seq_len = min(len(user_history), self.max_seq_len)
        
        # 2. Encode candidate items
        candidates = prompt.get("candidates", [])
        candidates_kjt = self._encode_candidates(candidates)
        num_cand = min(len(candidates), self.max_candidates)
        
        # 3. Build GRInputs
        sequence_lengths = torch.tensor([seq_len], dtype=torch.int32)
        candidate_counts = torch.tensor([num_cand], dtype=torch.int32)
        
        gr_inputs_data: GRInputs = {
            "type": "gr_inputs",
            "uih_features_kjt": uih_kjt,
            "candidates_features_kjt": candidates_kjt,
            "sequence_lengths": sequence_lengths,
            "candidate_counts": candidate_counts,
            "user_id": prompt["user_id"],
            "num_recommendations": num_recommendations,
        }
        
        if "context" in prompt:
            gr_inputs_data["context"] = prompt["context"]
        
        # 4. Create GRRequest (save original_candidates for output mapping)
        return GRRequest(
            request_id=request_id,
            gr_inputs=gr_inputs_data,
            user_id=prompt["user_id"],
            num_recommendations=num_recommendations,
            original_candidates=candidates,
            arrival_time=arrival_time,
            priority=priority,
        )
    
    def _encode_user_history(self, history: List[Dict]) -> Any:
        """
        Encode user history as KeyedJaggedTensor.
        
        Args:
            history: User history interaction list
            
        Returns:
            KeyedJaggedTensor or simplified dictionary format
        """
        # Take the most recent max_seq_len interactions
        recent_history = history[-self.max_seq_len:] if history else []
        
        # Extract item_id
        item_ids = [interaction.get("item_id", 0) for interaction in recent_history]
        
        # Use KeyedJaggedTensor if torchrec is available
        if _TORCHREC_AVAILABLE:
            return KeyedJaggedTensor(
                keys=self.uih_feature_names,
                values=torch.tensor(item_ids, dtype=torch.int64),
                lengths=torch.tensor([len(item_ids)], dtype=torch.int64),
            )
        
        # Otherwise return simplified format
        return {
            "keys": self.uih_feature_names,
            "values": item_ids,
            "lengths": [len(item_ids)],
        }
    
    def _encode_candidates(self, candidates: List[Dict]) -> Any:
        """
        Encode candidate items as KeyedJaggedTensor.
        
        Args:
            candidates: Candidate item list
            
        Returns:
            KeyedJaggedTensor or simplified dictionary format
        """
        # Take the first max_candidates candidates
        limited_candidates = candidates[:self.max_candidates] if candidates else []
        
        # Extract item_id
        item_ids = [item.get("item_id", 0) for item in limited_candidates]
        
        # Use KeyedJaggedTensor if torchrec is available
        if _TORCHREC_AVAILABLE:
            return KeyedJaggedTensor(
                keys=self.candidate_feature_names,
                values=torch.tensor(item_ids, dtype=torch.int64),
                lengths=torch.tensor([len(item_ids)], dtype=torch.int64),
            )
        
        # Otherwise return simplified format
        return {
            "keys": self.candidate_feature_names,
            "values": item_ids,
            "lengths": [len(item_ids)],
        }
    
    def get_item_ids_from_kjt(self, kjt: Any) -> List[int]:
        """
        Extract item_ids from KJT or simplified format.
        
        Args:
            kjt: KeyedJaggedTensor or simplified dictionary
            
        Returns:
            List of item_ids
        """
        if _TORCHREC_AVAILABLE and hasattr(kjt, 'values'):
            return kjt.values().tolist()
        elif isinstance(kjt, dict):
            return kjt.get("values", [])
        return []
