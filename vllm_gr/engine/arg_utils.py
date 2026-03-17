# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GREngineArgs: Engine arguments for generative recommendation.

Inherits from vLLM EngineArgs to maintain compatibility while adding
GR-specific parameters.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict
import os
import json
import glob
import logging

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

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
        max_seq_len: Maximum user history sequence length
        max_candidates: Maximum number of candidate items
        default_top_k: Default number of recommendations to return
    
    Feature names (uih, candidate, primary_id, task_names) are read
    directly from HSTUConfig (hf_config), which is the single source of
    truth.  Override via hf_overrides or config.json, not here.
    
    Model architecture parameters (num_layers, embedding_dim, etc.) live in
    config.json / HSTUConfig and are loaded via hf_config, following vLLM's
    convention.  They are NOT duplicated here.
    """
    
    # GR-specific: Sequence configuration
    max_seq_len: int = 512
    max_candidates: int = 1000
    
    # GR-specific: Output configuration
    default_top_k: int = 10
    
    # GR-specific: Dataset identifier (drives config overrides)
    dataset: Optional[str] = None
    
    def __post_init__(self):
        """Post-init processing to ensure proper configuration."""
        super().__post_init__()

        # Merge dataset overrides into hf_overrides (in-memory, never disk)
        self._apply_dataset_overrides()

        # Ensure trust_remote_code is enabled for HSTU models
        if self._is_hstu_model():
            self.trust_remote_code = True
            # Set max_model_len from max_seq_len if not explicitly set
            if self.max_model_len is None:
                self.max_model_len = self.max_seq_len
            # Ensure skip_tokenizer_init since GR models don't use tokenizers
            self.skip_tokenizer_init = True
            # Use runner="pooling" for recommendation models (task is deprecated)
            self.runner = "pooling"
            # Use eager mode for GR models (no CUDA graph optimization needed)
            self.enforce_eager = True
            
            # CRITICAL: Disable prefix caching for attention-free models
            # HSTU models don't use KV cache, so prefix caching must be disabled.
            self.enable_prefix_caching = False
            
            # Select the backend-specific GR worker implementation.
            self.worker_cls = self._select_gr_worker_cls()
            
            # HSTU models always use bfloat16 (matching upstream DLRM_v3)
            if self.dtype == "auto":
                self.dtype = "bfloat16"

            if self.tensor_parallel_size > 1:
                logger.warning(
                    "HSTU models use full model replication (data parallel), "
                    "not tensor sharding. Each worker loads the full model. "
                    "tensor_parallel_size=%d creates %d full replicas.",
                    self.tensor_parallel_size,
                    self.tensor_parallel_size,
                )

            # Determine weight loading strategy
            if not self._has_model_weights():
                self.load_format = "dummy"
                logger.info("No model weights found, using dummy loader (random weights)")

    def _is_ascend_platform(self) -> bool:
        """Return True when the active vLLM platform is Ascend/NPU."""
        return current_platform.device_type == "npu"

    def _select_gr_worker_cls(self) -> str:
        """Choose the GR worker implementation for the active platform."""
        if self._is_ascend_platform():
            return "vllm_gr.worker.gr_npu_worker.GRNPUWorker"
        return "vllm_gr.worker.gr_worker.GRWorker"
    
    def _has_model_weights(self) -> bool:
        """Check if the model path contains HuggingFace-style weight files."""
        if not self.model:
            return False

        if not os.path.isdir(self.model):
            return False

        hf_patterns = [
            "*.safetensors", "*.bin", "*.pt", "*.pth",
        ]
        for pattern in hf_patterns:
            if glob.glob(os.path.join(self.model, pattern)):
                return True

        return False
    
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
    
    def _apply_dataset_overrides(self) -> None:
        """Merge dataset-specific overrides into ``hf_overrides`` (in-memory).

        Unlike the previous implementation that mutated ``config.json`` on
        disk, this uses vLLM's ``hf_overrides`` mechanism so that overrides
        are applied in-memory when HuggingFace loads the config.  This is
        safe for concurrent processes and read-only model directories.

        User-supplied ``hf_overrides`` take precedence over dataset defaults.
        """
        if not self.dataset:
            return

        from vllm_gr.config.dataset_configs import (
            get_hstu_config_overrides,
            get_embedding_table_overrides,
        )

        overrides = get_hstu_config_overrides(self.dataset)
        tables = get_embedding_table_overrides(self.dataset)
        if tables:
            overrides["embedding_tables"] = tables

        if isinstance(self.hf_overrides, dict):
            merged = dict(overrides)
            merged.update(self.hf_overrides)
            self.hf_overrides = merged
        elif self.hf_overrides is None:
            self.hf_overrides = overrides
        logger.info("Applied dataset=%s overrides via hf_overrides", self.dataset)

    def create_engine_config(self, usage_context=None):
        """
        Create engine configuration for GR models.

        Explicitly initializes the current process via ``vllm_gr.plugin``
        before calling the parent implementation. Dataset overrides have
        already been merged into ``hf_overrides`` during ``__post_init__``.
        """
        from vllm_gr.plugin import initialize_runtime

        initialize_runtime()

        logger.info("Creating engine config for model: %s", self.model)
        return super().create_engine_config(usage_context)
