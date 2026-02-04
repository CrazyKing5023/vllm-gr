# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Top-K 采样器

根据模型预测分数选择得分最高的 K 个候选物品。
"""

from typing import List, Tuple, Optional

import torch


class TopKSampler:
    """
    Top-K 采样器
    
    根据预测分数选择 Top-K 候选
    """
    
    def __init__(self, task_weights: Optional[List[float]] = None):
        """
        初始化采样器
        
        Args:
            task_weights: 各任务的权重，用于计算综合分数
        """
        self.task_weights = task_weights
    
    def sample(
        self,
        predictions: torch.Tensor,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float, List[float]]]:
        """
        采样 Top-K 候选
        
        Args:
            predictions: (num_tasks, num_candidates) 预测分数
            candidate_ids: 候选 ID 列表
            top_k: 返回数量
        
        Returns:
            [(item_id, score, task_scores), ...]
        """
        # Handle 1D predictions (single task)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        num_tasks, num_candidates = predictions.shape
        
        # 确保 candidate_ids 长度匹配
        if len(candidate_ids) != num_candidates:
            raise ValueError(
                f"候选数量不匹配: predictions={num_candidates}, candidate_ids={len(candidate_ids)}"
            )
        
        # 计算综合分数
        if self.task_weights and len(self.task_weights) == num_tasks:
            weights = torch.tensor(self.task_weights, device=predictions.device, dtype=predictions.dtype)
            scores = (predictions * weights.unsqueeze(1)).sum(dim=0)
        else:
            # 默认取平均
            scores = predictions.mean(dim=0)
        
        # Top-K 选择
        k = min(top_k, num_candidates)
        topk_scores, topk_indices = torch.topk(scores, k)
        
        # 构建结果
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
        批量采样 Top-K 候选
        
        Args:
            batch_predictions: 每个批次元素的预测分数列表
            batch_candidate_ids: 每个批次元素的候选 ID 列表
            top_k: 每个元素返回的数量
        
        Returns:
            批量结果 [[(item_id, score, task_scores), ...], ...]
        """
        results = []
        for predictions, candidate_ids in zip(batch_predictions, batch_candidate_ids):
            results.append(self.sample(predictions, candidate_ids, top_k))
        return results

