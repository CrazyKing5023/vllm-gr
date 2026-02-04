# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HSTU 模型封装器

封装 DlrmHSTU 模型，提供简化的推理接口。
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, NamedTuple

import torch
import torch.nn as nn

from vllm_gr.config.hstu_config import HSTUModelConfig
from vllm_gr.model_executor.embedding_manager import EmbeddingManager
from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig, SequenceEmbedding


@dataclass
class HSTUModelOutput:
    """HSTU model output container."""
    predictions: torch.Tensor       # (num_tasks, num_candidates)
    user_embeddings: torch.Tensor   # (num_candidates, hidden_dim)
    item_embeddings: torch.Tensor   # (num_candidates, hidden_dim)


class HSTUModelWrapper(nn.Module):
    """
    HSTU 模型封装器
    
    封装移植后的 DlrmHSTU 模型，提供简化的推理接口。
    支持用户历史序列嵌入和候选物品嵌入的输入格式。
    """
    
    def __init__(
        self,
        config: HSTUModelConfig,
        embedding_manager: EmbeddingManager,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.embedding_manager = embedding_manager
        self.device = device
        
        # Create HSTU model
        self.hstu_model = self._create_model()
    
    def _create_model(self) -> DlrmHSTU:
        """Create the underlying DlrmHSTU model."""
        from vllm_gr.models.dlrm_v3.modules.multitask_module import TaskConfig, MultitaskTaskType
        
        # Convert config to DlrmHSTUConfig
        multitask_configs = []
        for i, task_name in enumerate(self.config.task_names):
            multitask_configs.append(TaskConfig(
                task_name=task_name,
                task_weight=1 << i,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ))
        
        hstu_config = DlrmHSTUConfig(
            max_seq_len=self.config.max_seq_len,
            max_num_candidates=self.config.max_num_candidates,
            hstu_num_heads=self.config.num_heads,
            hstu_attn_linear_dim=self.config.hidden_dim,
            hstu_attn_qk_dim=self.config.attention_dim,
            hstu_attn_num_layers=self.config.num_layers,
            hstu_embedding_table_dim=self.config.embedding_table_dim,
            hstu_transducer_embedding_dim=self.config.embedding_dim,
            hstu_preprocessor_hidden_dim=self.config.preprocessor_hidden_dim,
            hstu_input_dropout_ratio=self.config.input_dropout_ratio,
            hstu_linear_dropout_rate=self.config.linear_dropout_rate,
            multitask_configs=multitask_configs,
            user_embedding_feature_names=self.config.user_embedding_feature_names,
            item_embedding_feature_names=self.config.item_embedding_feature_names,
            uih_post_id_feature_name=self.config.uih_post_id_feature_name,
            uih_action_time_feature_name=self.config.uih_action_time_feature_name,
            uih_weight_feature_name=self.config.uih_weight_feature_name,
            candidates_weight_feature_name=self.config.candidates_weight_feature_name,
            candidates_watchtime_feature_name=self.config.candidates_watchtime_feature_name,
            candidates_querytime_feature_name=self.config.candidates_querytime_feature_name,
            merge_uih_candidate_feature_mapping=self.config.merge_uih_candidate_feature_mapping,
            contextual_feature_to_max_length=self.config.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=self.config.contextual_feature_to_min_uih_length,
            enable_postprocessor=self.config.enable_postprocessor,
            use_layer_norm_postprocessor=self.config.use_layer_norm_postprocessor,
        )
        
        return DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_collection=None,  # Embeddings managed separately
            is_inference=True,
            is_dense=True,  # Use dense mode (embeddings provided externally)
        )
    
    def forward(
        self,
        uih_embeddings: torch.Tensor,
        uih_lengths: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> HSTUModelOutput:
        """
        Forward pass for inference.
        
        Args:
            uih_embeddings: User interaction history embeddings (L_uih, D)
            uih_lengths: Length of each user's history (B,)
            candidate_embeddings: Candidate item embeddings (L_cand, D)
            num_candidates: Number of candidates per user (B,)
        
        Returns:
            HSTUModelOutput with predictions and embeddings
        """
        # Combine UIH and candidate embeddings
        total_embeddings = torch.cat([uih_embeddings, candidate_embeddings], dim=0)
        total_lengths = uih_lengths + num_candidates
        
        # Create sequence embeddings dict
        seq_embeddings = {
            "item_id": SequenceEmbedding(
                lengths=total_lengths,
                embedding=total_embeddings,
            ),
        }
        
        # Run model
        (
            user_embeddings,
            item_embeddings,
            aux_losses,
            predictions,
            _,
            _,
        ) = self.hstu_model(
            seq_embeddings=seq_embeddings,
            num_candidates=num_candidates,
        )
        
        # Ensure predictions have the right shape: (num_tasks, num_candidates)
        if predictions is None:
            predictions = torch.zeros(
                len(self.config.task_names), 
                int(num_candidates.sum().item()), 
                device=user_embeddings.device,
                dtype=user_embeddings.dtype,
            )
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        return HSTUModelOutput(
            predictions=predictions,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )
    
    def forward_simple(
        self,
        sequence_embeddings: torch.Tensor,
        sequence_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> HSTUModelOutput:
        """
        Simplified forward pass with combined sequence.
        
        Args:
            sequence_embeddings: Combined sequence embeddings (L, D)
            sequence_lengths: Total length per batch element (B,)
            num_candidates: Number of candidates per user (B,)
        
        Returns:
            HSTUModelOutput
        """
        seq_embeddings = {
            "item_id": SequenceEmbedding(
                lengths=sequence_lengths,
                embedding=sequence_embeddings,
            ),
        }
        
        (
            user_embeddings,
            item_embeddings,
            _,
            predictions,
            _,
            _,
        ) = self.hstu_model(
            seq_embeddings=seq_embeddings,
            num_candidates=num_candidates,
        )
        
        # Ensure predictions have the right shape: (num_tasks, num_candidates)
        if predictions is None:
            predictions = torch.zeros(
                len(self.config.task_names), 
                int(num_candidates.sum().item()), 
                device=user_embeddings.device,
                dtype=user_embeddings.dtype,
            )
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        return HSTUModelOutput(
            predictions=predictions,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )

