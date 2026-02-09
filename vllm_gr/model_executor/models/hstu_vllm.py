# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HSTU Model for vLLM-GR.

This is the unified HSTU model class that:
1. Is compatible with vLLM's ModelRegistry
2. Can be used directly for GR inference
3. Supports both HSTUConfig and HSTUModelConfig

Note: HSTU is a recommendation model (not LLM):
- Uses Target-Aware Attention instead of autoregressive generation
- Processes user history + candidates in a single forward pass
- Returns ranking scores instead of token probabilities
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HSTUModelOutput:
    """HSTU model output container."""
    predictions: torch.Tensor       # (num_tasks, num_candidates)
    user_embeddings: torch.Tensor   # (num_candidates, hidden_dim)
    item_embeddings: torch.Tensor   # (num_candidates, hidden_dim)


class HSTUForRecommendation(nn.Module):
    """
    Unified HSTU model for recommendation.
    
    This class provides:
    1. vLLM-compatible interface (forward, load_weights)
    2. GR inference interface (run_inference, run_recommendation)
    3. Support for both HSTUConfig and HSTUModelConfig
    
    Usage:
        # With HSTUConfig (HuggingFace-compatible)
        model = HSTUForRecommendation(config)
        
        # With HSTUModelConfig (internal)
        model = HSTUForRecommendation.from_model_config(model_config, device)
    """
    
    # vLLM compatibility flags
    supports_multimodal = False
    supports_pp = False
    
    def __init__(
        self,
        config: Any,  # HSTUConfig or HSTUModelConfig
        cache_config: Optional[Any] = None,
        quant_config: Optional[Any] = None,
        lora_config: Optional[Any] = None,
        prefix: str = "",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.prefix = prefix
        self._device = device
        
        # Normalize config to common attributes
        self.config = self._normalize_config(config)
        
        # Create the underlying HSTU model
        self._create_hstu_model()
        
        # Create embedding tables
        self._create_embedding_tables()
        
        logger.info(
            f"HSTUForRecommendation initialized: "
            f"layers={self.config.num_layers}, heads={self.config.num_heads}, "
            f"dim={self.config.embedding_dim}"
        )
    
    def _normalize_config(self, config: Any) -> Any:
        """Normalize config to ensure required attributes exist."""
        # Check if it's HSTUModelConfig (has more detailed fields)
        if hasattr(config, 'max_num_candidates'):
            # HSTUModelConfig uses max_num_candidates
            pass
        elif hasattr(config, 'max_candidates'):
            # Rename for consistency
            config.max_num_candidates = config.max_candidates
        else:
            config.max_num_candidates = 100
        
        # Ensure all required attributes
        defaults = {
            'num_layers': 5,
            'num_heads': 4,
            'embedding_dim': 512,
            'attention_dim': 128,
            'hidden_dim': 256,
            'embedding_table_dim': 192,
            'preprocessor_hidden_dim': 256,
            'max_seq_len': 1024,
            'max_num_candidates': 100,
            'hash_size': 10_000_000,
            'task_names': ['is_click'],
            'user_embedding_feature_names': ['item_id'],
            'item_embedding_feature_names': ['item_candidate_id'],
            'uih_post_id_feature_name': 'item_id',
            'input_dropout_ratio': 0.2,
            'linear_dropout_rate': 0.2,
            'embedding_tables': [],
        }
        
        for key, default in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, default)
        
        return config
    
    def _create_hstu_model(self) -> None:
        """Create the underlying DlrmHSTU model."""
        from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
        from vllm_gr.models.dlrm_v3.modules.multitask_module import TaskConfig, MultitaskTaskType
        
        # Build multitask configs
        multitask_configs = []
        for i, task_name in enumerate(self.config.task_names):
            multitask_configs.append(TaskConfig(
                task_name=task_name,
                task_weight=1 << i,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ))
        
        # Build DlrmHSTUConfig
        hstu_config_kwargs = {
            'max_seq_len': self.config.max_seq_len,
            'max_num_candidates': self.config.max_num_candidates,
            'hstu_num_heads': self.config.num_heads,
            'hstu_attn_linear_dim': self.config.hidden_dim,
            'hstu_attn_qk_dim': self.config.attention_dim,
            'hstu_attn_num_layers': self.config.num_layers,
            'hstu_embedding_table_dim': self.config.embedding_table_dim,
            'hstu_transducer_embedding_dim': self.config.embedding_dim,
            'hstu_preprocessor_hidden_dim': self.config.preprocessor_hidden_dim,
            'hstu_input_dropout_ratio': self.config.input_dropout_ratio,
            'hstu_linear_dropout_rate': self.config.linear_dropout_rate,
            'multitask_configs': multitask_configs,
            'user_embedding_feature_names': self.config.user_embedding_feature_names,
            'item_embedding_feature_names': self.config.item_embedding_feature_names,
            'uih_post_id_feature_name': self.config.uih_post_id_feature_name,
        }
        
        # Add optional fields if they exist
        optional_fields = [
            'uih_action_time_feature_name',
            'uih_weight_feature_name',
            'candidates_weight_feature_name',
            'candidates_watchtime_feature_name',
            'candidates_querytime_feature_name',
            'merge_uih_candidate_feature_mapping',
            'contextual_feature_to_max_length',
            'contextual_feature_to_min_uih_length',
            'enable_postprocessor',
            'use_layer_norm_postprocessor',
        ]
        for field in optional_fields:
            if hasattr(self.config, field):
                hstu_config_kwargs[field] = getattr(self.config, field)
        
        hstu_config = DlrmHSTUConfig(**hstu_config_kwargs)
        
        self.hstu = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_collection=None,
            is_inference=True,
            is_dense=True,
        )
    
    def _create_embedding_tables(self) -> None:
        """Create embedding tables."""
        self.embedding_tables = nn.ModuleDict()
        
        embedding_tables = getattr(self.config, 'embedding_tables', [])
        if not embedding_tables:
            # Create default embedding table
            self.embedding_tables['item_embedding'] = nn.Embedding(
                self.config.hash_size,
                self.config.embedding_table_dim,
            )
        else:
            for table_config in embedding_tables:
                if isinstance(table_config, dict):
                    name = table_config['name']
                    dim = table_config['embedding_dim']
                    num = table_config['num_embeddings']
                else:
                    name = table_config.name
                    dim = table_config.embedding_dim
                    num = table_config.num_embeddings
                self.embedding_tables[name] = nn.Embedding(num, dim)
    
    @classmethod
    def from_model_config(
        cls,
        config: Any,  # HSTUModelConfig
        device: torch.device,
        **kwargs,
    ) -> "HSTUForRecommendation":
        """Create model from HSTUModelConfig."""
        return cls(config=config, device=device, **kwargs)
    
    # =========================================================================
    # vLLM Interface
    # =========================================================================
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        attn_metadata: Optional[Any] = None,
        intermediate_tensors: Optional[Any] = None,
        # GR-specific inputs
        uih_embeddings: Optional[torch.Tensor] = None,
        uih_lengths: Optional[torch.Tensor] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
        num_candidates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass (vLLM compatible interface).
        
        Returns:
            Ranking scores (num_tasks, total_candidates)
        """
        if uih_embeddings is None or candidate_embeddings is None:
            logger.warning("forward() called without GR inputs, returning dummy")
            return torch.zeros(1, device=self._get_device())
        
        output = self._forward_impl(
            uih_embeddings, uih_lengths, candidate_embeddings, num_candidates
        )
        return output.predictions
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load model weights (vLLM interface)."""
        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                if param.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    loaded_params.add(name)
                else:
                    logger.warning(f"Shape mismatch for {name}")
        
        logger.info(f"Loaded {len(loaded_params)} parameters")
        return loaded_params
    
    # =========================================================================
    # GR Inference Interface
    # =========================================================================
    
    def run_inference(
        self,
        uih_embeddings: torch.Tensor,
        uih_lengths: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> HSTUModelOutput:
        """
        Run inference with embeddings.
        
        Args:
            uih_embeddings: User history embeddings (L_uih, D)
            uih_lengths: User history lengths (B,)
            candidate_embeddings: Candidate embeddings (L_cand, D)
            num_candidates: Candidates per user (B,)
        
        Returns:
            HSTUModelOutput with predictions and embeddings
        """
        return self._forward_impl(
            uih_embeddings, uih_lengths, candidate_embeddings, num_candidates
        )
    
    def run_recommendation(
        self,
        uih_item_ids: torch.Tensor,
        uih_lengths: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run recommendation with item IDs.
        
        Args:
            uih_item_ids: User history item IDs (L_uih,)
            uih_lengths: User history lengths (B,)
            candidate_item_ids: Candidate item IDs (L_cand,)
            num_candidates: Candidates per user (B,)
        
        Returns:
            Ranking scores (num_tasks, total_candidates)
        """
        # Lookup embeddings with modulo for hash-based addressing
        if "item_embedding" in self.embedding_tables:
            item_table = self.embedding_tables["item_embedding"]
        else:
            item_table = next(iter(self.embedding_tables.values()))
        
        # Apply modulo to ensure indices are within bounds
        num_embeddings = item_table.num_embeddings
        safe_uih_ids = uih_item_ids % num_embeddings
        safe_cand_ids = candidate_item_ids % num_embeddings
        
        uih_embeddings = item_table(safe_uih_ids)
        candidate_embeddings = item_table(safe_cand_ids)
        
        output = self._forward_impl(
            uih_embeddings, uih_lengths, candidate_embeddings, num_candidates
        )
        return output.predictions
    
    def lookup_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for item IDs.
        
        Applies modulo operation to ensure item IDs are within embedding table bounds.
        This is standard practice in recommendation systems for hash-based embeddings.
        """
        if "item_embedding" in self.embedding_tables:
            item_table = self.embedding_tables["item_embedding"]
        else:
            item_table = next(iter(self.embedding_tables.values()))
        
        # Apply modulo to ensure indices are within bounds
        num_embeddings = item_table.num_embeddings
        safe_ids = item_ids % num_embeddings
        
        return item_table(safe_ids)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _forward_impl(
        self,
        uih_embeddings: torch.Tensor,
        uih_lengths: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> HSTUModelOutput:
        """Internal forward implementation."""
        from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import SequenceEmbedding
        
        # Combine UIH and candidate embeddings
        total_embeddings = torch.cat([uih_embeddings, candidate_embeddings], dim=0)
        total_lengths = uih_lengths + num_candidates
        
        # Build seq_embeddings dict
        seq_embeddings: Dict[str, SequenceEmbedding] = {}
        
        for name in self.config.user_embedding_feature_names:
            seq_embeddings[name] = SequenceEmbedding(
                lengths=total_lengths,
                embedding=total_embeddings,
            )
        
        for name in self.config.item_embedding_feature_names:
            seq_embeddings[name] = SequenceEmbedding(
                lengths=num_candidates,
                embedding=candidate_embeddings,
            )
        
        # Ensure uih_post_id_feature_name is available
        if self.config.uih_post_id_feature_name not in seq_embeddings:
            seq_embeddings[self.config.uih_post_id_feature_name] = SequenceEmbedding(
                lengths=total_lengths,
                embedding=total_embeddings,
            )
        
        # Run HSTU model
        (
            user_embeddings,
            item_embeddings,
            aux_losses,
            predictions,
            _,
            _,
        ) = self.hstu(
            seq_embeddings=seq_embeddings,
            num_candidates=num_candidates,
        )
        
        # Ensure predictions shape
        if predictions is None:
            predictions = torch.zeros(
                len(self.config.task_names),
                int(num_candidates.sum().item()),
                device=self._get_device(),
            )
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        return HSTUModelOutput(
            predictions=predictions,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )
    
    def _get_device(self) -> torch.device:
        """Get model device."""
        if self._device is not None:
            return self._device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
