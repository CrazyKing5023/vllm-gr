# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HuggingFace-compatible HSTU Configuration.

This module provides a PretrainedConfig subclass for HSTU models,
enabling config.json parsing compatible with HuggingFace/vLLM patterns.

This module registers HSTUConfig with HuggingFace's AutoConfig so that:
1. AutoConfig.from_pretrained("path/to/hstu-model") works correctly
2. vLLM's engine config creation can load HSTU models
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import transformers PretrainedConfig if available
try:
    from transformers import PretrainedConfig, AutoConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    PretrainedConfig = object  # type: ignore
    AutoConfig = None  # type: ignore


class HSTUConfig(PretrainedConfig if _HAS_TRANSFORMERS else object):
    """
    HuggingFace-compatible configuration for HSTU models.
    
    This class parses config.json files with model_type="hstu".
    
    Example config.json:
        {
            "model_type": "hstu",
            "architectures": ["HSTUEmbeddingModel"],
            "num_layers": 8,
            "num_heads": 8,
            "embedding_dim": 512,
            ...
        }
    
    Note: The architecture name "HSTUEmbeddingModel" follows vLLM's convention
    for embedding models (suffix "EmbeddingModel" -> pooling runner).
    """
    
    model_type = "hstu"
    # Default architectures - use EmbeddingModel suffix for vLLM compatibility
    architectures = ["HSTUEmbeddingModel"]
    
    # vLLM pooling model indicators - these mark the model as an embedding/pooling model
    # so that vLLM uses the pooling runner instead of the generate runner
    is_encoder_decoder = False
    is_text_encoder = True  # Indicates this is an encoder-only model (embedding model)
    
    # HSTU doesn't use standard attention mechanism / KV cache
    # This tells vLLM to skip KV cache setup
    is_attention_free = True
    
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
        # vLLM pooling model flags (can be overridden)
        is_text_encoder: bool = True,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        if _HAS_TRANSFORMERS:
            super().__init__(**kwargs)
        
        # Set vLLM pooling model flags
        self.is_text_encoder = is_text_encoder
        self.is_encoder_decoder = is_encoder_decoder
        
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


# Register HSTUConfig with HuggingFace's AutoConfig
# This allows AutoConfig.from_pretrained() to recognize model_type="hstu"

_HSTU_CONFIG_REGISTERED = False


def _register_hstu_config() -> bool:
    """
    Register HSTUConfig with HuggingFace AutoConfig.
    
    Returns:
        True if registration succeeded or already registered
    """
    global _HSTU_CONFIG_REGISTERED
    
    if _HSTU_CONFIG_REGISTERED:
        return True
    
    if not _HAS_TRANSFORMERS or AutoConfig is None:
        logger.warning("transformers not available, cannot register HSTUConfig")
        return False
    
    try:
        # Register with AutoConfig - this is the standard way in transformers
        AutoConfig.register("hstu", HSTUConfig)
        _HSTU_CONFIG_REGISTERED = True
        logger.info("Registered HSTUConfig with HuggingFace AutoConfig")
        return True
    except ValueError as e:
        # "hstu is already used by another config" - this is fine, already registered
        if "already used" in str(e).lower():
            _HSTU_CONFIG_REGISTERED = True
            logger.debug("HSTUConfig already registered with AutoConfig")
            return True
        else:
            logger.error(f"Failed to register HSTUConfig: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to register HSTUConfig: {e}")
        return False


def ensure_hstu_config_registered() -> bool:
    """
    Ensure HSTUConfig is registered with HuggingFace AutoConfig.
    
    Call this function before any vLLM config loading to ensure
    the 'hstu' model_type is recognized.
    
    Returns:
        True if HSTUConfig is registered
    """
    return _register_hstu_config()


def is_hstu_config_registered() -> bool:
    """Check if HSTUConfig is registered with AutoConfig."""
    return _HSTU_CONFIG_REGISTERED


# Auto-register when module is imported
_register_hstu_config()
