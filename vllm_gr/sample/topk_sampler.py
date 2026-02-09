# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Top-K Sampler

Selects the top K candidates with highest scores based on model predictions.
"""

from typing import List, Tuple, Optional

import torch


class TopKSampler:
    """
    Top-K Sampler
    
    Selects Top-K candidates based on prediction scores.
    """
    
    def __init__(self, task_weights: Optional[List[float]] = None):
        """
        Initialize sampler
        
        Args:
            task_weights: Weights for each task, used to compute combined score
        """
        self.task_weights = task_weights
    
    def sample(
        self,
        predictions: torch.Tensor,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float, List[float]]]:
        """
        Sample Top-K candidates
        
        Args:
            predictions: (num_tasks, num_candidates) prediction scores
            candidate_ids: Candidate ID list
            top_k: Number to return
        
        Returns:
            [(item_id, score, task_scores), ...]
        """
        # Handle 1D predictions (single task)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        num_tasks, num_candidates = predictions.shape
        
        # Ensure candidate_ids length matches
        if len(candidate_ids) != num_candidates:
            raise ValueError(
                f"Candidate count mismatch: predictions={num_candidates}, candidate_ids={len(candidate_ids)}"
            )
        
        # Compute combined score
        if self.task_weights and len(self.task_weights) == num_tasks:
            weights = torch.tensor(self.task_weights, device=predictions.device, dtype=predictions.dtype)
            scores = (predictions * weights.unsqueeze(1)).sum(dim=0)
        else:
            # Default to average
            scores = predictions.mean(dim=0)
        
        # Top-K selection
        k = min(top_k, num_candidates)
        topk_scores, topk_indices = torch.topk(scores, k)
        
        # Build results
        results = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            task_scores = predictions[:, idx].tolist()
            results.append((candidate_ids[idx], score, task_scores))
        
        return results
    
    def sample_batch(
        self,
        batch_predictions: List[torch.Tensor],
        batch_candidate_ids: List[List[str]],
        top_k: int,
    ) -> List[List[Tuple[str, float, List[float]]]]:
        """
        Batch sample Top-K candidates
        
        Args:
            batch_predictions: List of prediction scores for each batch element
            batch_candidate_ids: List of candidate IDs for each batch element
            top_k: Number to return for each element
        
        Returns:
            Batch results [[(item_id, score, task_scores), ...], ...]
        """
        results = []
        for predictions, candidate_ids in zip(batch_predictions, batch_candidate_ids):
            results.append(self.sample(predictions, candidate_ids, top_k))
        return results
