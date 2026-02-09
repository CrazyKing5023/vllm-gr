# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GREngineArgs: Engine arguments for generative recommendation.

Extends vLLM EngineArgs with recommendation-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List

# vLLM imports - mandatory dependency (vLLM 0.11.0)
from vllm.engine.arg_utils import EngineArgs


@dataclass
class GREngineArgs(EngineArgs):
    """
    GR Engine Arguments.
    
    Inherits from vLLM EngineArgs and adds recommendation engine-specific parameters.
    """
    
    # Model configuration (required field inherited from EngineArgs)
    model: str = ""
    """Model name or path"""
    
    # GR-specific parameters
    embedding_path: Optional[str] = None
    """Embedding table path (optional, for loading pretrained embeddings)"""
    
    max_item_id: int = 10000
    """Maximum item ID (corresponds to dataset's max_item_id)"""
    
    # Sequence configuration
    max_seq_len: int = 512
    """Maximum user history sequence length"""
    
    max_candidates: int = 1000
    """Maximum number of candidate items"""
    
    # Output configuration
    default_top_k: int = 10
    """Default number of recommendations"""
    
    # Feature configuration (corresponds to generative-recommenders)
    uih_feature_names: List[str] = field(default_factory=lambda: ["item_id"])
    """User history feature name list"""
    
    candidate_feature_names: List[str] = field(default_factory=lambda: ["candidate_item_id"])
    """Candidate item feature name list"""
    
    # Model loading configuration
    device: Optional[str] = None
    """Inference device (cuda/cpu), None for auto-selection"""
    
    embedding_dim: int = 192
    """Embedding vector dimension"""
    
    hash_size: int = 10_000_000
    """Embedding table hash size"""
    
    load_model: bool = True
    """Whether to load and initialize the HSTU model"""
    
    use_vllm_registry: bool = False
    """Whether to use vLLM ModelRegistry for HSTU model loading.
    
    If True: Uses vLLM's standard model loading with registered HSTUForRecommendation.
             Requires model_type="hstu" and architectures=["HSTUForRecommendation"] in config.json.
    If False (default): Uses GR-native model loading (HSTUModelConfig + GRModelRunner).
                        More flexible, does not require HuggingFace-compatible config.
    """
    
    # HSTU model architecture parameters (used when creating model from config without pretrained weights)
    num_layers: int = 3
    """Number of HSTU layers"""
    
    num_heads: int = 2
    """Number of attention heads"""
    
    hidden_dim: int = 128
    """Hidden layer dimension"""
    
    attention_dim: int = 64
    """Attention dimension"""
    
    def __post_init__(self):
        """Validate parameters."""
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive, got {self.max_candidates}")
        if self.default_top_k <= 0:
            raise ValueError(f"default_top_k must be positive, got {self.default_top_k}")
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        
        # Call parent's __post_init__ if it exists
        # Use try-except because vLLM's validation may fail for HSTU (non-LLM model)
        if hasattr(super(), '__post_init__'):
            try:
                super().__post_init__()
            except Exception:
                # Skip parent validation for HSTU models (vLLM expects LLM-style config)
                pass
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "embedding_path": self.embedding_path,
            "max_item_id": self.max_item_id,
            "max_seq_len": self.max_seq_len,
            "max_candidates": self.max_candidates,
            "default_top_k": self.default_top_k,
            "uih_feature_names": self.uih_feature_names,
            "candidate_feature_names": self.candidate_feature_names,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "hash_size": self.hash_size,
            "load_model": self.load_model,
            "use_vllm_registry": self.use_vllm_registry,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "attention_dim": self.attention_dim,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "disable_log_stats": self.disable_log_stats,
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "GREngineArgs":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})
