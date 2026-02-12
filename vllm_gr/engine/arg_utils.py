# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GREngineArgs: Engine arguments for generative recommendation.

Inherits from vLLM EngineArgs to maintain compatibility while adding
GR-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
import os
import json
import glob
import logging

from vllm.engine.arg_utils import EngineArgs

logger = logging.getLogger(__name__)


@dataclass
class GREngineArgs(EngineArgs):
    """
    GR Engine Arguments - Inherits from vLLM EngineArgs.
    
    Extends vLLM's EngineArgs with GR-specific parameters while maintaining
    full compatibility with vLLM's configuration system.
    
    Inherited from EngineArgs (commonly used):
        model: Model name or path
        tensor_parallel_size: Tensor parallelism degree
        gpu_memory_utilization: GPU memory utilization fraction
        disable_log_stats: Whether to disable logging statistics
        max_model_len: Maximum model length (maps to max_seq_len for GR)
        
    GR-specific parameters:
        device: Inference device (cuda/cpu), None for auto-detect
        embedding_path: Path to pretrained embeddings
        max_seq_len: Maximum user history sequence length
        max_candidates: Maximum number of candidate items
        default_top_k: Default number of recommendations to return
        embedding_dim: Embedding vector dimension
        hash_size: Embedding table hash size
        load_model: Whether to load the model
        num_layers: Number of HSTU layers
        num_heads: Number of attention heads
        hidden_dim: Hidden layer dimension
        attention_dim: Attention dimension
        use_scheduler: Whether to use scheduler
        gr_max_batch_size: Maximum batch size for GR scheduler
        scheduling_policy: Scheduling policy (fcfs, priority)
        uih_feature_names: User history feature names
        candidate_feature_names: Candidate feature names
    """
    
    # GR-specific: Device configuration
    device: Optional[str] = None  # cuda, cpu, or None for auto-detect
    
    # GR-specific: Embedding configuration
    embedding_path: Optional[str] = None
    embedding_dim: int = 192
    hash_size: int = 10_000_000
    
    # GR-specific: Sequence configuration
    max_seq_len: int = 512
    max_candidates: int = 1000
    
    # GR-specific: Output configuration
    default_top_k: int = 10
    
    # GR-specific: Model loading
    load_model: bool = True
    
    # GR-specific: HSTU architecture parameters
    num_layers: int = 3
    num_heads: int = 2
    hidden_dim: int = 128
    attention_dim: int = 64
    
    # GR-specific: Scheduler configuration
    use_scheduler: bool = True
    gr_max_batch_size: int = 32  # Renamed to avoid conflict with parent
    scheduling_policy: str = "fcfs"
    
    # GR-specific: Feature configuration
    uih_feature_names: List[str] = field(default_factory=lambda: ["item_id"])
    candidate_feature_names: List[str] = field(default_factory=lambda: ["candidate_item_id"])
    
    # GR-specific: Execution mode
    force_gr_native: bool = False  # If True, skip EngineCore and use GR-native mode
    
    def __post_init__(self):
        """Post-init processing to ensure proper configuration."""
        # Ensure trust_remote_code is enabled for HSTU models
        if self._is_hstu_model():
            self.trust_remote_code = True
            # Set max_model_len from max_seq_len if not explicitly set
            if self.max_model_len is None:
                self.max_model_len = self.max_seq_len
            # Ensure skip_tokenizer_init since GR models don't use tokenizers
            self.skip_tokenizer_init = True
            # Use runner="pooling" for recommendation models (task is deprecated)
            # This tells vLLM to use the pooling runner instead of generate runner
            self.runner = "pooling"
            # Use eager mode for GR models (no CUDA graph optimization needed)
            self.enforce_eager = True
            
            # Check if model has real weights, if not use dummy loader
            # This allows EngineCore to work without real weight files
            if not self._has_model_weights():
                self.load_format = "dummy"
                logger.info("No model weights found, using dummy loader (random weights)")
    
    def _has_model_weights(self) -> bool:
        """Check if the model path contains actual weight files."""
        if not self.model:
            return False
        
        if not os.path.isdir(self.model):
            return False
        
        # Check for common weight file patterns
        weight_patterns = [
            "*.bin",           # PyTorch checkpoint
            "*.safetensors",   # Safetensors format
            "*.pt",            # PyTorch state dict
            "*.pth",           # PyTorch model
            "model.safetensors",
            "pytorch_model.bin",
        ]
        
        for pattern in weight_patterns:
            if glob.glob(os.path.join(self.model, pattern)):
                return True
        
        return False
    
    def to_gr_dict(self) -> Dict[str, Any]:
        """Convert GR-specific parameters to dictionary."""
        return {
            "device": self.device,
            "embedding_path": self.embedding_path,
            "max_seq_len": self.max_seq_len,
            "max_candidates": self.max_candidates,
            "default_top_k": self.default_top_k,
            "embedding_dim": self.embedding_dim,
            "hash_size": self.hash_size,
            "load_model": self.load_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "attention_dim": self.attention_dim,
            "use_scheduler": self.use_scheduler,
            "gr_max_batch_size": self.gr_max_batch_size,
            "scheduling_policy": self.scheduling_policy,
            "uih_feature_names": self.uih_feature_names,
            "candidate_feature_names": self.candidate_feature_names,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GREngineArgs":
        """Create from dictionary."""
        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    def _is_hstu_model(self) -> bool:
        """Check if the model is an HSTU model."""
        if not self.model:
            return True  # No model path = will create HSTU with random weights
        
        if os.path.isdir(self.model):
            config_path = os.path.join(self.model, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    model_type = config.get("model_type", "")
                    return model_type.lower() == "hstu"
                except Exception:
                    pass
        return True  # Default to HSTU
    
    def create_engine_config(self, usage_context=None):
        """
        Create engine configuration for GR models.
        
        Ensures HSTUConfig is registered with HuggingFace AutoConfig before
        calling the parent's create_engine_config() to use EngineCore.
        """
        # CRITICAL: Ensure HSTUConfig is registered BEFORE any config loading
        from vllm_gr.config.hstu_hf_config import ensure_hstu_config_registered
        ensure_hstu_config_registered()
        
        # Also ensure GR models are registered with vLLM ModelRegistry
        from vllm_gr.model_executor.models.registry import register_gr_models_to_vllm
        register_gr_models_to_vllm()
        
        logger.info(f"Creating engine config for model: {self.model}")
        
        # Call parent's implementation which will use EngineCore properly
        vllm_config = super().create_engine_config(usage_context)
        
        # Ensure skip_tokenizer_init for GR (tokenizer not needed)
        if hasattr(vllm_config, 'model_config') and vllm_config.model_config is not None:
            vllm_config.model_config.skip_tokenizer_init = True
        
        return vllm_config
