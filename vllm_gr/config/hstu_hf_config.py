# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""HuggingFace-compatible HSTU configuration and vLLM integration."""

from typing import Any, Dict, List, Optional
import logging

from transformers import PretrainedConfig, AutoConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

logger = logging.getLogger(__name__)


class HSTUConfig(PretrainedConfig):
    """PretrainedConfig subclass for HSTU models (model_type="hstu")."""

    model_type = "hstu"
    architectures = ["HSTUEmbeddingModel"]
    is_encoder_decoder = False
    is_text_encoder = True
    is_attention_free = True

    def __init__(
        self,
        num_layers: int = 5,
        num_heads: int = 4,
        embedding_dim: int = 512,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        embedding_table_dim: int = 192,
        preprocessor_hidden_dim: int = 256,
        max_seq_len: int = 1024,
        max_num_candidates: int = 100,
        hash_size: int = 10_000_000,
        feature_names: Optional[List[str]] = None,
        user_embedding_feature_names: Optional[List[str]] = None,
        item_embedding_feature_names: Optional[List[str]] = None,
        uih_post_id_feature_name: str = "item_id",
        task_names: Optional[List[str]] = None,
        embedding_tables: Optional[List[Dict[str, Any]]] = None,
        input_dropout_ratio: float = 0.2,
        linear_dropout_rate: float = 0.2,
        is_inference: bool = True,
        is_text_encoder: bool = True,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_text_encoder = is_text_encoder
        self.is_encoder_decoder = is_encoder_decoder

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_table_dim = embedding_table_dim
        self.preprocessor_hidden_dim = preprocessor_hidden_dim

        self.max_seq_len = max_seq_len
        self.max_num_candidates = max_num_candidates

        self.hash_size = hash_size
        self.feature_names = feature_names or ["user_id", "item_id", "item_candidate_id"]
        self.user_embedding_feature_names = user_embedding_feature_names or ["item_id"]
        self.item_embedding_feature_names = item_embedding_feature_names or ["item_candidate_id"]
        self.uih_post_id_feature_name = uih_post_id_feature_name

        self.task_names = task_names or ["is_click"]
        self.embedding_tables = embedding_tables or [
            {
                "name": "item_embedding",
                "embedding_dim": embedding_table_dim,
                "num_embeddings": hash_size,
                "feature_names": ["item_id", "item_candidate_id"],
            }
        ]

        self.input_dropout_ratio = input_dropout_ratio
        self.linear_dropout_rate = linear_dropout_rate
        self.is_inference = is_inference


# Maps HSTU-native field names (embedding_dim, num_layers, …) to the
# standard names that vLLM's ModelArchitectureConfig expects.
class HSTUModelArchConfigConvertor(ModelArchConfigConvertorBase):

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "embedding_dim", 0)

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_layers", 0)

    def get_total_num_attention_heads(self) -> int:
        return getattr(self.hf_text_config, "num_heads", 0)

    def get_head_size(self) -> int:
        return getattr(self.hf_text_config, "attention_dim", 0)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "hash_size", 0)

    def get_total_num_kv_heads(self) -> int:
        return 0  # attention-free

    def derive_max_model_len_and_key(self) -> tuple:
        max_seq = getattr(self.hf_text_config, "max_seq_len", 1024)
        return float(max_seq), "max_seq_len"


# Registration ----------------------------------------------------------------

_HSTU_CONFIG_REGISTERED = False


def _register_hstu_config() -> bool:
    global _HSTU_CONFIG_REGISTERED
    if _HSTU_CONFIG_REGISTERED:
        return True

    try:
        AutoConfig.register("hstu", HSTUConfig)
        _HSTU_CONFIG_REGISTERED = True
    except ValueError as e:
        if "already used" in str(e).lower():
            _HSTU_CONFIG_REGISTERED = True
        else:
            logger.error("Failed to register HSTUConfig: %s", e)
            return False

    if "hstu" not in MODEL_ARCH_CONFIG_CONVERTORS:
        MODEL_ARCH_CONFIG_CONVERTORS["hstu"] = HSTUModelArchConfigConvertor

    return True


def ensure_hstu_config_registered() -> bool:
    return _register_hstu_config()


def is_hstu_config_registered() -> bool:
    return _HSTU_CONFIG_REGISTERED


_register_hstu_config()
