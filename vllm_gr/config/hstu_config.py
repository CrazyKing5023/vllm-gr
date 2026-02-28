# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HSTU Model Configuration

Defines model architecture parameters, embedding table configuration, and multitask settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import IntEnum
import json
import os


class MultitaskTaskType(IntEnum):
    """Multitask type enumeration"""
    BINARY_CLASSIFICATION = 0
    REGRESSION = 1


@dataclass
class TaskConfig:
    """Task configuration"""
    task_name: str
    task_weight: int
    task_type: MultitaskTaskType


@dataclass
class EmbeddingTableConfig:
    """Embedding table configuration"""
    name: str
    embedding_dim: int
    num_embeddings: int
    feature_names: List[str]
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbeddingTableConfig":
        return cls(**d)


@dataclass
class HSTUModelConfig:
    """HSTU Model Configuration"""
    
    # Model architecture
    num_layers: int = 5
    num_heads: int = 4
    embedding_dim: int = 512
    attention_dim: int = 128
    hidden_dim: int = 256
    embedding_table_dim: int = 192
    preprocessor_hidden_dim: int = 256
    
    # Sequence parameters
    max_seq_len: int = 1024
    max_num_candidates: int = 100
    
    # Feature configuration
    hash_size: int = 10_000_000
    feature_names: List[str] = field(default_factory=lambda: [
        "user_id", "item_id", "item_candidate_id"
    ])
    
    # User/Item feature names
    user_embedding_feature_names: List[str] = field(default_factory=lambda: ["item_id"])
    item_embedding_feature_names: List[str] = field(default_factory=lambda: ["item_candidate_id"])
    uih_post_id_feature_name: str = "item_id"
    uih_action_time_feature_name: str = "action_time"
    uih_weight_feature_name: str = "action_weight"
    candidates_weight_feature_name: str = "candidate_weight"
    candidates_watchtime_feature_name: str = "candidate_watchtime"
    candidates_querytime_feature_name: str = "query_time"
    
    # Feature mapping
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(default_factory=list)
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    context_feature_names: List[str] = field(
        default_factory=lambda: ["query_id", "scenario_id"]
    )
    
    # Multitask configuration
    task_names: List[str] = field(default_factory=lambda: ["is_click", "is_like"])
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    causal_multitask_weights: float = 0.2
    
    # Embedding table configuration
    embedding_tables: List[EmbeddingTableConfig] = field(default_factory=list)
    
    # Dropout
    input_dropout_ratio: float = 0.2
    linear_dropout_rate: float = 0.2
    
    # Postprocessor configuration
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False
    
    # Inference mode
    is_inference: bool = True
    
    def __post_init__(self):
        """Post-initialization processing"""
        # If no multitask config, create default config based on task_names
        if not self.multitask_configs and self.task_names:
            self.multitask_configs = [
                TaskConfig(
                    task_name=name,
                    task_weight=1 << i,
                    task_type=MultitaskTaskType.BINARY_CLASSIFICATION
                )
                for i, name in enumerate(self.task_names)
            ]
        
        # If no embedding tables configured, create default configuration
        if not self.embedding_tables:
            self.embedding_tables = [
                EmbeddingTableConfig(
                    name="item_embedding",
                    embedding_dim=self.embedding_table_dim,
                    num_embeddings=self.hash_size,
                    feature_names=["item_id", "item_candidate_id"]
                ),
                EmbeddingTableConfig(
                    name="user_embedding",
                    embedding_dim=self.embedding_table_dim,
                    num_embeddings=self.hash_size // 10,
                    feature_names=["user_id"]
                )
            ]
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "HSTUModelConfig":
        """Load configuration from model path"""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        else:
            # Return default configuration
            return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HSTUModelConfig":
        """Create configuration from dictionary"""
        # Process embedding table configuration
        if "embedding_tables" in config_dict:
            config_dict["embedding_tables"] = [
                EmbeddingTableConfig.from_dict(t) if isinstance(t, dict) else t
                for t in config_dict["embedding_tables"]
            ]
        
        # Process multitask configuration
        if "multitask_configs" in config_dict:
            config_dict["multitask_configs"] = [
                TaskConfig(**t) if isinstance(t, dict) else t
                for t in config_dict["multitask_configs"]
            ]
        
        # Filter unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "embedding_dim": self.embedding_dim,
            "attention_dim": self.attention_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_table_dim": self.embedding_table_dim,
            "max_seq_len": self.max_seq_len,
            "max_num_candidates": self.max_num_candidates,
            "hash_size": self.hash_size,
            "feature_names": self.feature_names,
            "task_names": self.task_names,
            "is_inference": self.is_inference,
        }
    
    def save(self, path: str) -> None:
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
