# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Input Processor

Converts JSON format recommendation requests to tensor format acceptable by the model.
"""

from typing import List, Dict, Any, Tuple, Optional

import torch

from vllm_gr.config.hstu_config import HSTUModelConfig


class GRInputProcessor:
    """
    Input Processor
    
    Converts JSON format recommendation requests to model input tensors.
    Supports feature hashing for user behavior sequences and candidate items.
    """
    
    def __init__(self, config: HSTUModelConfig):
        self.config = config
        self.feature_names = config.feature_names
        self.hash_size = config.hash_size
    
    def process(
        self,
        user_id: str,
        sequence: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input request
        
        Args:
            user_id: User ID
            sequence: User behavior sequence [{"item_id": ..., "timestamp": ...}, ...]
            candidates: Candidate items list [{"item_id": ...}, ...]
            device: Target device
        
        Returns:
            (sequence_indices, sequence_lengths, candidate_indices, num_candidates)
        """
        device = device or torch.device("cpu")
        
        # 1. Process user behavior sequence
        sequence_indices = self._process_sequence(sequence)
        sequence_lengths = torch.tensor([len(sequence_indices)], dtype=torch.long, device=device)
        
        # 2. Process candidate items
        candidate_indices = self._process_candidates(candidates)
        num_candidates = torch.tensor([len(candidate_indices)], dtype=torch.long, device=device)
        
        # Move to device
        sequence_indices = sequence_indices.to(device)
        candidate_indices = candidate_indices.to(device)
        
        return sequence_indices, sequence_lengths, candidate_indices, num_candidates
    
    def process_batch(
        self,
        requests: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch process requests
        
        Args:
            requests: Request list, each request contains user_id, sequence, candidates
            device: Target device
        
        Returns:
            (all_sequence_indices, sequence_lengths, all_candidate_indices, num_candidates)
        """
        device = device or torch.device("cpu")
        
        all_sequence_indices = []
        sequence_lengths = []
        all_candidate_indices = []
        num_candidates_list = []
        
        for req in requests:
            seq_indices = self._process_sequence(req.get("sequence", []))
            cand_indices = self._process_candidates(req.get("candidates", []))
            
            all_sequence_indices.append(seq_indices)
            sequence_lengths.append(len(seq_indices))
            all_candidate_indices.append(cand_indices)
            num_candidates_list.append(len(cand_indices))
        
        # Concatenate all sequences
        combined_sequence = torch.cat(all_sequence_indices, dim=0) if all_sequence_indices else torch.tensor([], dtype=torch.long)
        combined_candidates = torch.cat(all_candidate_indices, dim=0) if all_candidate_indices else torch.tensor([], dtype=torch.long)
        
        sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long, device=device)
        num_candidates = torch.tensor(num_candidates_list, dtype=torch.long, device=device)
        
        return (
            combined_sequence.to(device),
            sequence_lengths,
            combined_candidates.to(device),
            num_candidates,
        )
    
    def _process_sequence(self, sequence: List[Dict[str, Any]]) -> torch.Tensor:
        """Process user behavior sequence"""
        if not sequence:
            return torch.tensor([], dtype=torch.long)
        
        indices = []
        for item in sequence:
            item_id = item.get("item_id", "")
            idx = self._hash_feature("item_id", item_id)
            indices.append(idx)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _process_candidates(self, candidates: List[Dict[str, Any]]) -> torch.Tensor:
        """Process candidate items"""
        if not candidates:
            return torch.tensor([], dtype=torch.long)
        
        indices = []
        for item in candidates:
            item_id = item.get("item_id", "")
            idx = self._hash_feature("item_id", item_id)
            indices.append(idx)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _hash_feature(self, feature_name: str, value: Any) -> int:
        """
        Hash feature value
        
        Args:
            feature_name: Feature name
            value: Feature value
        
        Returns:
            Hashed index
        """
        if isinstance(value, int):
            return value % self.hash_size
        
        # Use Python hash for string or other types
        return hash(f"{feature_name}:{value}") % self.hash_size
