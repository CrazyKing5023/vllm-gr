# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
输出处理器

将模型预测结果转换为 JSON 格式的推荐响应。
"""

from typing import List, Dict, Any, Tuple

from vllm_gr.config.hstu_config import HSTUModelConfig


class GROutputProcessor:
    """
    输出处理器
    
    将采样后的推荐结果格式化为 JSON 响应。
    支持多任务分数的输出。
    """
    
    def __init__(self, config: HSTUModelConfig):
        self.config = config
    
    def format_output(
        self,
        sampled: List[Tuple[str, float, List[float]]],
        task_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        格式化输出
        
        Args:
            sampled: [(item_id, score, task_scores), ...]
            task_names: 任务名称列表
        
        Returns:
            推荐结果列表
        """
        recommendations = []
        
        for rank, (item_id, score, task_scores) in enumerate(sampled):
            rec = {
                "item_id": item_id,
                "score": round(float(score), 4),
                "rank": rank,
            }
            
            # 添加各任务分数
            if task_names and task_scores:
                rec["task_scores"] = {
                    name: round(float(ts), 4)
                    for name, ts in zip(task_names, task_scores)
                }
            
            recommendations.append(rec)
        
        return recommendations
    
    def format_batch_output(
        self,
        batch_sampled: List[List[Tuple[str, float, List[float]]]],
        task_names: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """
        批量格式化输出
        
        Args:
            batch_sampled: 批量采样结果
            task_names: 任务名称列表
        
        Returns:
            批量推荐结果
        """
        return [
            self.format_output(sampled, task_names)
            for sampled in batch_sampled
        ]

