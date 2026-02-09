# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HuggingFace-compatible HSTU Configuration.

This module provides a PretrainedConfig subclass for HSTU models,
enabling config.json parsing compatible with HuggingFace/vLLM patterns.

Note: vLLM uses ModelRegistry (not AutoConfig) for model resolution.
The model_type="hstu" in config.json is parsed by this class, but
model loading is handled by GRModelRegistry.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import transformers PretrainedConfig if available
try:
    from transformers import PretrainedConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    PretrainedConfig = object  # type: ignore


class HSTUConfig(PretrainedConfig if _HAS_TRANSFORMERS else object):
    """
    HuggingFace-compatible configuration for HSTU models.
    
    This class parses config.json files with model_type="hstu".
    
    Example config.json:
        {
            "model_type": "hstu",
            "architectures": ["HSTUForRecommendation"],
            "num_layers": 8,
            "num_heads": 8,
            "embedding_dim": 512,
            ...
        }
    """
    
    model_type = "hstu"
    
    def __init__(
        self,
        # Model architecture
        num_layers: int = 5,
        num_heads: int = 4,
        embedding_dim: int = 512,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        embedding_table_dim: int = 192,
        preprocessor_hidden_dim: int = 256,
        # Sequence parameters
        max_seq_len: int = 1024,
        max_num_candidates: int = 100,
        # Feature configuration
        hash_size: int = 10_000_000,
        feature_names: Optional[List[str]] = None,
        user_embedding_feature_names: Optional[List[str]] = None,
        item_embedding_feature_names: Optional[List[str]] = None,
        uih_post_id_feature_name: str = "item_id",
        # Multitask configuration
        task_names: Optional[List[str]] = None,
        # Embedding tables
        embedding_tables: Optional[List[Dict[str, Any]]] = None,
        # Dropout
        input_dropout_ratio: float = 0.2,
        linear_dropout_rate: float = 0.2,
        # Inference mode
        is_inference: bool = True,
        **kwargs,
    ):
        if _HAS_TRANSFORMERS:
            super().__init__(**kwargs)
        
        # Model architecture
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_table_dim = embedding_table_dim
        self.preprocessor_hidden_dim = preprocessor_hidden_dim
        
        # Sequence parameters
        self.max_seq_len = max_seq_len
        self.max_num_candidates = max_num_candidates
        
        # Feature configuration
        self.hash_size = hash_size
        self.feature_names = feature_names or ["user_id", "item_id", "item_candidate_id"]
        self.user_embedding_feature_names = user_embedding_feature_names or ["item_id"]
        self.item_embedding_feature_names = item_embedding_feature_names or ["item_candidate_id"]
        self.uih_post_id_feature_name = uih_post_id_feature_name
        
        # Multitask configuration
        self.task_names = task_names or ["is_click"]
        
        # Embedding tables
        self.embedding_tables = embedding_tables or [
            {
                "name": "item_embedding",
                "embedding_dim": embedding_table_dim,
                "num_embeddings": hash_size,
                "feature_names": ["item_id", "item_candidate_id"]
            }
        ]
        
        # Dropout
        self.input_dropout_ratio = input_dropout_ratio
        self.linear_dropout_rate = linear_dropout_rate
        
        # Inference mode
        self.is_inference = is_inference
