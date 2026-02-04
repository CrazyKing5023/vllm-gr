# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Output Processor

Converts model prediction results to JSON format recommendation responses.
"""

from typing import List, Dict, Any, Tuple

from vllm_gr.config.hstu_config import HSTUModelConfig


class GROutputProcessor:
    """
    Output Processor
    
    Formats sampled recommendation results into JSON responses.
    Supports multi-task score output.
    """
    
    def __init__(self, config: HSTUModelConfig):
        self.config = config
    
    def format_output(
        self,
        sampled: List[Tuple[str, float, List[float]]],
        task_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Format output
        
        Args:
            sampled: [(item_id, score, task_scores), ...]
            task_names: Task name list
        
        Returns:
            Recommendation results list
        """
        recommendations = []
        
        for rank, (item_id, score, task_scores) in enumerate(sampled):
            rec = {
                "item_id": item_id,
                "score": round(float(score), 4),
                "rank": rank,
            }
            
            # Add individual task scores
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
        Batch format output
        
        Args:
            batch_sampled: Batch sampling results
            task_names: Task name list
        
        Returns:
            Batch recommendation results
        """
        return [
            self.format_output(sampled, task_names)
            for sampled in batch_sampled
        ]
