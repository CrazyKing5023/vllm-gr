# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
HSTU Model for vLLM-GR.

Unified HSTU model class that:
1. Is compatible with vLLM's ModelRegistry
2. Delegates embedding lookup and preprocess to DlrmHSTU (via EmbeddingCollection)
3. Accepts KeyedJaggedTensor inputs from GRModelRunner

HSTU is a recommendation model (not LLM):
- Uses Target-Aware Attention instead of autoregressive generation
- Processes user history + candidates in a single forward pass
- Returns ranking scores instead of token probabilities
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HSTUPooler(nn.Module):
    """
    Pooler module for HSTU model — provides vLLM pooling model compatibility.

    For recommendation models, the "pooling" operation is essentially
    returning the item scores/predictions rather than text embeddings.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Identity()

    def get_supported_tasks(self) -> set:
        """Valid PoolingTask values: "embed", "classify", "score", etc."""
        return {"embed"}

    def get_pooling_updates(self, task: str) -> Any:
        from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
        return PoolingParamsUpdate()

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        pooling_metadata: Optional[Any] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(hidden_states, list):
            return [self.dense(h) for h in hidden_states]
        return self.dense(hidden_states)


class HSTUForRecommendation(nn.Module):
    """
    Unified HSTU model for recommendation.

    vLLM-compatible wrapper around DlrmHSTU.  DlrmHSTU owns the
    EmbeddingCollection internally and handles preprocess + forward.
    This class only:
      - translates HSTUConfig → DlrmHSTUConfig + EmbeddingConfig
      - provides the vLLM forward / load_weights interface
    """

    supports_multimodal = False
    supports_pp = False
    is_pooling_model = True
    is_text_generation_model = False
    is_attention_free = True

    def __init__(
        self,
        *,
        vllm_config: Any = None,
        prefix: str = "",
        config: Any = None,
        cache_config: Any = None,
        quant_config: Any = None,
        lora_config: Any = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.prefix = prefix
        self._device = device
        self._device_checked = False

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config

        self.config = config

        self._create_hstu_model()

        self._pooler = HSTUPooler(self.config.hstu_transducer_embedding_dim)

        logger.info(
            "HSTUForRecommendation initialized: layers=%d, heads=%d, dim=%d",
            self.config.hstu_attn_num_layers,
            self.config.hstu_num_heads,
            self.config.hstu_transducer_embedding_dim,
        )

    @property
    def pooler(self):
        return self._pooler

    def _build_embedding_configs(self) -> Dict[str, "EmbeddingConfig"]:
        """Convert HSTUConfig.embedding_tables (list of dicts) to TorchRec EmbeddingConfig dict."""
        from torchrec.modules.embedding_configs import EmbeddingConfig, DataType

        configs: Dict[str, EmbeddingConfig] = {}
        for table_cfg in self.config.embedding_tables:
            name = table_cfg["name"]
            configs[name] = EmbeddingConfig(
                num_embeddings=table_cfg["num_embeddings"],
                embedding_dim=table_cfg["embedding_dim"],
                name=name,
                data_type=DataType.FP32,
                feature_names=table_cfg.get("feature_names", [name]),
            )
        return configs

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_hstu_model(self) -> None:
        """Create the underlying DlrmHSTU model with EmbeddingCollection."""
        from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import (
            DlrmHSTU, DlrmHSTUConfig,
        )
        from vllm_gr.models.dlrm_v3.modules.multitask_module import (
            TaskConfig, MultitaskTaskType,
        )

        multitask_configs = []
        for i, task_name in enumerate(self.config.task_names):
            multitask_configs.append(TaskConfig(
                task_name=task_name,
                task_weight=1 << i,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ))

        merge_mapping = [tuple(pair) for pair in self.config.merge_uih_candidate_feature_mapping]

        hstu_config = DlrmHSTUConfig(
            max_seq_len=self.config.max_seq_len,
            max_num_candidates=self.config.max_num_candidates,
            max_num_candidates_inference=self.config.max_num_candidates_inference,
            hstu_num_heads=self.config.hstu_num_heads,
            hstu_attn_linear_dim=self.config.hstu_attn_linear_dim,
            hstu_attn_qk_dim=self.config.hstu_attn_qk_dim,
            hstu_attn_num_layers=self.config.hstu_attn_num_layers,
            hstu_embedding_table_dim=self.config.hstu_embedding_table_dim,
            hstu_preprocessor_hidden_dim=self.config.hstu_preprocessor_hidden_dim,
            hstu_transducer_embedding_dim=self.config.hstu_transducer_embedding_dim,
            hstu_group_norm=self.config.hstu_group_norm,
            hstu_input_dropout_ratio=self.config.hstu_input_dropout_ratio,
            hstu_linear_dropout_rate=self.config.hstu_linear_dropout_rate,
            contextual_feature_to_max_length=self.config.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=self.config.contextual_feature_to_min_uih_length,
            candidates_weight_feature_name=self.config.candidates_weight_feature_name,
            candidates_watchtime_feature_name=self.config.candidates_watchtime_feature_name,
            candidates_querytime_feature_name=self.config.candidates_querytime_feature_name,
            multitask_configs=multitask_configs,
            user_embedding_feature_names=self.config.user_embedding_feature_names,
            item_embedding_feature_names=self.config.item_embedding_feature_names,
            uih_post_id_feature_name=self.config.uih_post_id_feature_name,
            uih_action_time_feature_name=self.config.uih_action_time_feature_name,
            uih_weight_feature_name=self.config.uih_weight_feature_name,
            hstu_uih_feature_names=self.config.hstu_uih_feature_names,
            hstu_candidate_feature_names=self.config.hstu_candidate_feature_names,
            merge_uih_candidate_feature_mapping=merge_mapping,
            action_weights=self.config.action_weights or None,
        )

        embedding_configs = self._build_embedding_configs()

        self.hstu = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=embedding_configs,
            is_inference=self.config.is_inference,
            is_dense=not embedding_configs,
        )

    # ------------------------------------------------------------------
    # vLLM interface
    # ------------------------------------------------------------------

    def _ensure_on_device(self, target_device: torch.device) -> None:
        """Move all tensors to target_device if needed (safety for vLLM buffers)."""
        moved = 0
        for _, buf in self.named_buffers():
            if buf.device != target_device:
                buf.data = buf.data.to(target_device)
                moved += 1
        for _, param in self.named_parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)
                moved += 1
        if moved:
            logger.warning("Moved %d tensors to %s", moved, target_device)
        self._device_checked = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        uih_features: Optional[Any] = None,
        candidates_features: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM-compatible forward pass.

        When uih_features and candidates_features (KeyedJaggedTensors) are
        provided, runs the full DlrmHSTU inference pipeline.  Otherwise
        returns a dummy tensor for vLLM warm-up.
        """
        if uih_features is None or candidates_features is None:
            total_tokens = input_ids.numel() if input_ids is not None else 1
            device = input_ids.device if input_ids is not None else self._get_device()
            return torch.zeros(
                total_tokens, self.config.hstu_transducer_embedding_dim,
                device=device,
            )

        if not self._device_checked:
            self._ensure_on_device(uih_features.values().device)

        (
            _user_emb, _item_emb, _aux,
            predictions, _labels, _weights,
        ) = self.hstu(uih_features, candidates_features)

        if predictions is None:
            predictions = torch.zeros(
                1, 1, device=self._get_device(),
            )
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        return predictions

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
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
                    logger.warning("Shape mismatch for %s", name)

        logger.info("Loaded %d parameters", len(loaded_params))
        return loaded_params

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        device: torch.device,
        **kwargs,
    ) -> "HSTUForRecommendation":
        return cls(config=config, device=device, **kwargs)

    def _get_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
