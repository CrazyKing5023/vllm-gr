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

from vllm_gr.config.hstu_config import (
    get_hstu_contextual_seq_len,
    get_hstu_full_sequence_len,
)

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
        max_model_len: Maximum scheduler-visible model length
        
    GR-specific parameters:
        max_seq_len: Maximum user history sequence length
        max_candidates: Maximum number of candidate items
    
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
    
    # GR-specific: Dataset identifier (drives config overrides)
    dataset: Optional[str] = None

    # GR-specific: explicit model family for backend/runtime defaults.
    # Keep this separate from HuggingFace's ``model_type`` so future GR
    # model families can coexist without assuming unknown models are HSTU.
    model_family: Optional[str] = None

    # GR-specific: explicit opt-in for demo/example flows that intentionally
    # provide only config.json and rely on randomly initialized weights.
    # Keep the default strict so local model directories without weights fail
    # fast unless the caller clearly requests dummy loading.
    allow_missing_weights: bool = False
    
    def __post_init__(self):
        """Post-init processing to ensure proper configuration."""
        super().__post_init__()

        # Merge dataset overrides into hf_overrides (in-memory, never disk)
        self._apply_dataset_overrides()

        self.model_family = self._resolve_model_family()
        self._apply_model_family_defaults()

    def _is_ascend_platform(self) -> bool:
        """Return True when the active vLLM platform is Ascend/NPU."""
        return current_platform.device_type == "npu"

    def _select_hstu_worker_cls(self) -> str:
        """Choose the HSTU GR worker implementation for the active platform."""
        if self._is_ascend_platform():
            return "vllm_gr.worker.gr_npu_worker.GRNPUWorker"
        return "vllm_gr.worker.gr_worker.GRWorker"

    def _select_worker_cls_for_model_family(
        self,
        model_family: Optional[str],
    ) -> Optional[str]:
        """Choose a worker only when the model family defines one."""
        if model_family == "hstu":
            return self._select_hstu_worker_cls()
        return None
    
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
    
    def _normalize_model_family(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    def _read_model_type_from_hf_overrides(self) -> Optional[str]:
        if isinstance(self.hf_overrides, dict):
            model_type = self.hf_overrides.get("model_type")
            if isinstance(model_type, str):
                return self._normalize_model_family(model_type)
        return None

    def _read_model_type_from_local_config(self) -> Optional[str]:
        config = self._read_local_config_dict()
        if config is None:
            return None
        model_type = config.get("model_type")
        if isinstance(model_type, str):
            return self._normalize_model_family(model_type)
        return None

    def _read_local_config_dict(self) -> Optional[Dict[str, Any]]:
        if not self.model or not os.path.isdir(self.model):
            return None

        config_path = os.path.join(self.model, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read local model config at {config_path}: {exc}"
            ) from exc

    def _resolve_model_family(self) -> Optional[str]:
        """Resolve the GR model family without assuming unknown == HSTU."""
        explicit = self._normalize_model_family(self.model_family)
        if explicit is not None:
            return explicit

        override_model_type = self._read_model_type_from_hf_overrides()
        if override_model_type is not None:
            return override_model_type

        config_model_type = self._read_model_type_from_local_config()
        if config_model_type is not None:
            return config_model_type

        if not self.model:
            # The in-repo no-path flow is still the current HSTU bootstrap path.
            return "hstu"

        return None

    def _apply_model_family_defaults(self) -> None:
        """Apply runtime defaults only for known GR model families."""
        if self.model_family != "hstu":
            return

        self.trust_remote_code = True
        required_max_model_len = self._get_hstu_runtime_max_model_len()
        if self.max_model_len is None:
            self.max_model_len = required_max_model_len
        elif self.max_model_len < required_max_model_len:
            raise ValueError(
                "For HSTU, max_model_len must cover the scheduler-visible full "
                "sequence: contextual prefix + UIH/history + candidates. "
                f"Got max_model_len={self.max_model_len}, but the current "
                f"runtime settings require at least {required_max_model_len} "
                f"(max_seq_len={self.max_seq_len}, "
                f"max_candidates={self.max_candidates})."
            )
        self.skip_tokenizer_init = True
        self.runner = "pooling"
        self.enforce_eager = True
        self._ensure_hstu_model_max_length_override(required_max_model_len)

        # HSTU models don't use KV cache, so prefix caching must be disabled.
        self.enable_prefix_caching = False

        worker_cls = self._select_worker_cls_for_model_family(self.model_family)
        if worker_cls is not None:
            self.worker_cls = worker_cls

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

        if self.model and os.path.isdir(self.model) and not self._has_model_weights():
            if self.allow_missing_weights or self.load_format == "dummy":
                self.load_format = "dummy"
                logger.info(
                    "No model weights found in local model directory %r; "
                    "using dummy loader because allow_missing_weights=True",
                    self.model,
                )
                return

            raise FileNotFoundError(
                "No model weight files found in local model directory "
                f"{self.model!r}. Expected one of *.safetensors, *.bin, *.pt, *.pth. "
                "If this is an intentional demo/random-init run, pass "
                "allow_missing_weights=True explicitly."
            )

    def _resolve_hstu_contextual_seq_len(self) -> int:
        if isinstance(self.hf_overrides, dict):
            contextual_seq_len = get_hstu_contextual_seq_len(self.hf_overrides)
            if contextual_seq_len > 0:
                return contextual_seq_len

        local_config = self._read_local_config_dict()
        if local_config is not None:
            return get_hstu_contextual_seq_len(local_config)
        return 0

    def _get_hstu_runtime_max_model_len(self) -> int:
        return get_hstu_full_sequence_len(
            max_seq_len=self.max_seq_len,
            max_candidates=self.max_candidates,
            contextual_seq_len=self._resolve_hstu_contextual_seq_len(),
        )

    def _ensure_hstu_model_max_length_override(self, max_model_len: int) -> None:
        if isinstance(self.hf_overrides, dict):
            current_value = self.hf_overrides.get("model_max_length")
            if current_value is None or int(current_value) < max_model_len:
                self.hf_overrides["model_max_length"] = max_model_len
            return

        if self.hf_overrides is None:
            self.hf_overrides = {"model_max_length": max_model_len}
            return

        if callable(self.hf_overrides):
            original_overrides = self.hf_overrides

            def _wrapped_hf_overrides(config):
                updated = original_overrides(config)
                target = updated if updated is not None else config
                current_value = getattr(target, "model_max_length", None)
                if current_value is None or int(current_value) < max_model_len:
                    setattr(target, "model_max_length", max_model_len)
                return updated

            self.hf_overrides = _wrapped_hf_overrides
    
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
