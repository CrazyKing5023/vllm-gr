# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HSTU Model Wrapper

Wraps the DlrmHSTU model and provides a simplified inference interface.
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
    HSTU Model Wrapper
    
    Wraps the transplanted DlrmHSTU model and provides a simplified inference interface.
    Supports user interaction history embeddings and candidate item embeddings as input format.
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
    
    def _build_seq_embeddings(
        self,
        total_embeddings: torch.Tensor,
        total_lengths: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> Dict[str, SequenceEmbedding]:
        """
        Build sequence embeddings dict with all required feature names.
        
        This ensures that both user_embedding_feature_names and item_embedding_feature_names
        are properly populated in the seq_embeddings dict, matching the expected format
        from the original generative-recommenders implementation.
        
        Args:
            total_embeddings: Combined UIH + candidate embeddings
            total_lengths: Total lengths per batch element
            candidate_embeddings: Candidate item embeddings only
            num_candidates: Number of candidates per batch element
        
        Returns:
            Dictionary mapping feature names to SequenceEmbedding objects
        """
        seq_embeddings: Dict[str, SequenceEmbedding] = {}
        
        # Add user embedding features (use combined sequence)
        for name in self.config.user_embedding_feature_names:
            seq_embeddings[name] = SequenceEmbedding(
                lengths=total_lengths,
                embedding=total_embeddings,
            )
        
        # Add item embedding features (use candidate embeddings only)
        for name in self.config.item_embedding_feature_names:
            seq_embeddings[name] = SequenceEmbedding(
                lengths=num_candidates,
                embedding=candidate_embeddings,
            )
        
        # Ensure uih_post_id_feature_name is available (needed by _user_forward)
        if self.config.uih_post_id_feature_name and self.config.uih_post_id_feature_name not in seq_embeddings:
            seq_embeddings[self.config.uih_post_id_feature_name] = SequenceEmbedding(
                lengths=total_lengths,
                embedding=total_embeddings,
            )
        
        return seq_embeddings
    
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
        
        # Create sequence embeddings dict with all required feature names
        # The key must match the feature names defined in config
        seq_embeddings = self._build_seq_embeddings(
            total_embeddings=total_embeddings,
            total_lengths=total_lengths,
            candidate_embeddings=candidate_embeddings,
            num_candidates=num_candidates,
        )
        
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
        # Extract candidate embeddings from the combined sequence
        # Candidate embeddings are at the end of the sequence
        total_candidates = int(num_candidates.sum().item())
        if total_candidates > 0:
            candidate_embeddings = sequence_embeddings[-total_candidates:]
        else:
            candidate_embeddings = sequence_embeddings.new_empty(0, sequence_embeddings.size(-1))
        
        # Build seq_embeddings with all required feature names
        seq_embeddings = self._build_seq_embeddings(
            total_embeddings=sequence_embeddings,
            total_lengths=sequence_lengths,
            candidate_embeddings=candidate_embeddings,
            num_candidates=num_candidates,
        )
        
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
