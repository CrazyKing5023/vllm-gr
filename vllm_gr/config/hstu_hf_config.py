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
    """PretrainedConfig subclass for HSTU models (model_type="hstu").

    Field names align with Meta's DlrmHSTUConfig from generative-recommenders.
    """

    model_type = "hstu"
    architectures = ["HSTUEmbeddingModel"]
    is_encoder_decoder = False
    is_text_encoder = True
    is_attention_free = True

    def __init__(
        self,
        # --- Sequence / candidate limits (Meta DlrmHSTUConfig order) ---
        max_seq_len: int = 1024,
        max_num_candidates: int = 100,
        max_num_candidates_inference: int = 5,
        # --- HSTU architecture ---
        hstu_num_heads: int = 4,
        hstu_attn_linear_dim: int = 256,
        hstu_attn_qk_dim: int = 128,
        hstu_attn_num_layers: int = 5,
        hstu_embedding_table_dim: int = 192,
        hstu_preprocessor_hidden_dim: int = 256,
        hstu_transducer_embedding_dim: int = 512,
        hstu_group_norm: bool = False,
        hstu_input_dropout_ratio: float = 0.2,
        hstu_linear_dropout_rate: float = 0.2,
        # --- Feature configs ---
        contextual_feature_to_max_length: Optional[Dict[str, int]] = None,
        contextual_feature_to_min_uih_length: Optional[Dict[str, int]] = None,
        candidates_weight_feature_name: str = "",
        candidates_watchtime_feature_name: str = "",
        candidates_querytime_feature_name: str = "",
        user_embedding_feature_names: Optional[List[str]] = None,
        item_embedding_feature_names: Optional[List[str]] = None,
        uih_post_id_feature_name: str = "item_id",
        uih_action_time_feature_name: str = "",
        uih_weight_feature_name: str = "",
        hstu_uih_feature_names: Optional[List[str]] = None,
        hstu_candidate_feature_names: Optional[List[str]] = None,
        merge_uih_candidate_feature_mapping: Optional[List[List[str]]] = None,
        action_weights: Optional[List[int]] = None,
        action_embedding_init_std: float = 0.1,
        causal_multitask_weights: float = 0.2,
        enable_postprocessor: bool = True,
        use_layer_norm_postprocessor: bool = False,
        # --- vllm-GR specific (not in Meta DlrmHSTUConfig) ---
        hash_size: int = 10_000_000,
        task_names: Optional[List[str]] = None,
        embedding_tables: Optional[List[Dict[str, Any]]] = None,
        is_inference: bool = True,
        is_text_encoder: bool = True,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.architectures = ["HSTUEmbeddingModel"]

        # Upstream DLRM_v3 always uses bfloat16 for inference.
        if "torch_dtype" not in kwargs:
            self.torch_dtype = "bfloat16"

        self.is_text_encoder = is_text_encoder
        self.is_encoder_decoder = is_encoder_decoder
        self.is_attention_free = True

        self.max_seq_len = max_seq_len
        self.max_num_candidates = max_num_candidates
        self.max_num_candidates_inference = max_num_candidates_inference

        self.hstu_num_heads = hstu_num_heads
        self.hstu_attn_linear_dim = hstu_attn_linear_dim
        self.hstu_attn_qk_dim = hstu_attn_qk_dim
        self.hstu_attn_num_layers = hstu_attn_num_layers
        self.hstu_embedding_table_dim = hstu_embedding_table_dim
        self.hstu_preprocessor_hidden_dim = hstu_preprocessor_hidden_dim
        self.hstu_transducer_embedding_dim = hstu_transducer_embedding_dim
        self.hstu_group_norm = hstu_group_norm
        self.hstu_input_dropout_ratio = hstu_input_dropout_ratio
        self.hstu_linear_dropout_rate = hstu_linear_dropout_rate

        self.contextual_feature_to_max_length = contextual_feature_to_max_length or {}
        self.contextual_feature_to_min_uih_length = contextual_feature_to_min_uih_length or {}
        self.candidates_weight_feature_name = candidates_weight_feature_name
        self.candidates_watchtime_feature_name = candidates_watchtime_feature_name
        self.candidates_querytime_feature_name = candidates_querytime_feature_name

        self.user_embedding_feature_names = user_embedding_feature_names or ["item_id"]
        self.item_embedding_feature_names = item_embedding_feature_names or ["item_candidate_id"]
        self.uih_post_id_feature_name = uih_post_id_feature_name
        self.uih_action_time_feature_name = uih_action_time_feature_name
        self.uih_weight_feature_name = uih_weight_feature_name
        self.hstu_uih_feature_names = hstu_uih_feature_names or []
        self.hstu_candidate_feature_names = hstu_candidate_feature_names or []
        self.merge_uih_candidate_feature_mapping = merge_uih_candidate_feature_mapping or []
        self.action_weights = action_weights or []
        self.action_embedding_init_std = action_embedding_init_std
        self.causal_multitask_weights = causal_multitask_weights
        self.enable_postprocessor = enable_postprocessor
        self.use_layer_norm_postprocessor = use_layer_norm_postprocessor

        self.hash_size = hash_size
        self.task_names = task_names or ["is_click"]
        self.embedding_tables = embedding_tables or [
            {
                "name": "item_embedding",
                "embedding_dim": hstu_embedding_table_dim,
                "num_embeddings": hash_size,
                "feature_names": ["item_id", "item_candidate_id"],
            }
        ]
        self.is_inference = is_inference


class HSTUModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Maps HSTUConfig fields to vLLM's ModelArchitectureConfig interface."""

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hstu_transducer_embedding_dim

    def get_num_hidden_layers(self) -> int:
        return self.hf_text_config.hstu_attn_num_layers

    def get_total_num_attention_heads(self) -> int:
        return self.hf_text_config.hstu_num_heads

    def get_head_size(self) -> int:
        return self.hf_text_config.hstu_attn_qk_dim

    def get_vocab_size(self) -> int:
        return self.hf_text_config.hash_size

    def get_total_num_kv_heads(self) -> int:
        return 0  # attention-free

    def derive_max_model_len_and_key(self) -> tuple:
        return float(self.hf_text_config.max_seq_len), "max_seq_len"


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
