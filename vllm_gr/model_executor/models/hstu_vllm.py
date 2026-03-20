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

Device placement:
- Sparse (embedding) weights live on CPU to handle large tables without GPU OOM
- Dense (MLP/Attention) weights live on GPU for fast computation
- Forward pass: preprocess(CPU) → move_sparse_output_to_device → main_forward(GPU)
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def move_sparse_output_to_device(
    seq_embeddings: Dict[str, Any],
    payload_features: Dict[str, torch.Tensor],
    uih_seq_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    device: torch.device,
) -> Tuple[
    Dict[str, Any],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Move sparse module (CPU) outputs to GPU for dense forward pass.

    Converts embedding outputs to bfloat16 for efficient GPU computation.
    """
    from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import SequenceEmbedding

    num_candidates = num_candidates.to(device)
    uih_seq_lengths = uih_seq_lengths.to(device)
    seq_embeddings = {
        k: SequenceEmbedding(
            lengths=seq_embeddings[k].lengths.to(device),
            embedding=seq_embeddings[k].embedding.to(device).to(torch.bfloat16),
        )
        for k in seq_embeddings.keys()
    }
    for k, v in payload_features.items():
        payload_features[k] = v.to(device)
    return seq_embeddings, payload_features, uih_seq_lengths, num_candidates


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

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config

        self.config = config

        # Determine GPU device for dense forward pass.
        # During vLLM model loading, the default device context is set to the
        # target GPU. We capture it here for move_sparse_output_to_device.
        if device is not None:
            self._gpu_device = device
        elif torch.cuda.is_available():
            self._gpu_device = torch.device(torch.cuda.current_device())
        else:
            self._gpu_device = torch.device("cpu")

        self._create_hstu_model()

        self._pooler = HSTUPooler(self.config.hstu_transducer_embedding_dim)

        logger.info(
            "HSTUForRecommendation initialized: layers=%d, heads=%d, dim=%d, "
            "gpu_device=%s, embedding_device=cpu",
            self.config.hstu_attn_num_layers,
            self.config.hstu_num_heads,
            self.config.hstu_transducer_embedding_dim,
            self._gpu_device,
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
                data_type=DataType.FP16,
                feature_names=table_cfg.get("feature_names", [name]),
            )
        return configs

    def _cast_dense_submodules_to_bf16(self) -> None:
        """Cast dense HSTU submodules to bf16 without touching embeddings."""
        dense_submodule_names = (
            "_multitask_module",
            "_hstu_transducer",
            "_item_embedding_mlp",
        )

        casted = []
        for name in dense_submodule_names:
            module = getattr(self.hstu, name, None)
            if module is None:
                logger.warning(
                    "Skipping dense bf16 cast: HSTU submodule %s not found",
                    name,
                )
                continue
            module.to(dtype=torch.bfloat16)
            casted.append(name)

        self.hstu.set_training_dtype(torch.bfloat16)
        logger.info(
            "Cast dense HSTU submodules to bf16: %s; embedding_collection left unchanged",
            ", ".join(casted) if casted else "<none>",
        )

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
            action_embedding_init_std=getattr(self.config, "action_embedding_init_std", 0.1),
            causal_multitask_weights=getattr(self.config, "causal_multitask_weights", 0.2),
            enable_postprocessor=getattr(self.config, "enable_postprocessor", True),
            use_layer_norm_postprocessor=getattr(self.config, "use_layer_norm_postprocessor", False),
        )

        embedding_configs = self._build_embedding_configs()

        self.hstu = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=embedding_configs,
            is_inference=self.config.is_inference,
            is_dense=not embedding_configs,
        )
        self._cast_dense_submodules_to_bf16()
        
    # ------------------------------------------------------------------
    # vLLM interface
    # ------------------------------------------------------------------

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
        """Two-phase forward pass with sparse/dense device separation.

        Phase 1 (CPU): preprocess() — embedding lookups via EmbeddingCollection
                       on CPU. Produces seq_embeddings + payload_features.
        Transfer:      move_sparse_output_to_device() — CPU → GPU.
        Phase 2 (GPU): main_forward() — attention, MLPs, multitask prediction
                       on GPU.

        When uih_features/candidates_features are None (vLLM warmup),
        returns a zero tensor on GPU.
        """
        if uih_features is None or candidates_features is None:
            total_tokens = input_ids.numel() if input_ids is not None else 1
            return torch.zeros(
                total_tokens, self.config.hstu_transducer_embedding_dim,
                device=self._gpu_device,
            )

        # Phase 1: sparse preprocess on CPU (embedding lookups)
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self.hstu.preprocess(
            uih_features=uih_features,
            candidates_features=candidates_features,
        )

        # Transfer: CPU → GPU
        seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
            move_sparse_output_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=self._gpu_device,
            )
        )

        # Phase 2: dense forward on GPU
        (
            _user_emb, _item_emb, _aux,
            predictions, _labels, _weights,
        ) = self.hstu.main_forward(
            seq_embeddings=seq_embeddings,
            payload_features=payload_features,
            max_uih_len=max_uih_len,
            uih_seq_lengths=uih_seq_lengths,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
        )

        if predictions is None:
            predictions = torch.zeros(
                1, 1, device=self._gpu_device,
            )
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        return predictions

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
        """Load model weights with sparse/dense device separation.

        - Sparse weights (embedding_collection) stay on CPU
        - Dense weights go to GPU
        - Registered buffers are loaded manually (AutoWeightsLoader
          only handles nn.Module and nn.Parameter)

        Device placement is automatic: parameters were created on their
        target device (CPU for embeddings via explicit device="cpu",
        GPU for everything else via vLLM's default device context).
        param.data.copy_() copies to wherever the parameter already lives.
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader

        buffer_names = {name for name, _ in self.named_buffers()}

        buffer_weights: List[Tuple[str, torch.Tensor]] = []
        param_weights: List[Tuple[str, torch.Tensor]] = []
        sparse_count = 0
        dense_count = 0

        for name, tensor in weights:
            is_sparse = "embedding_collection" in name
            if name in buffer_names:
                buffer_weights.append((name, tensor))
            else:
                param_weights.append((name, tensor))

            if is_sparse:
                sparse_count += 1
            else:
                dense_count += 1

        loaded: Set[str] = set()

        if buffer_weights:
            buf_dict = dict(self.named_buffers())
            for name, tensor in buffer_weights:
                buf_dict[name].copy_(tensor)
                loaded.add(name)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["_pooler."],
        )
        loaded |= loader.load_weights(iter(param_weights))

        logger.info(
            "Loaded %d weights: %d sparse (CPU), %d dense (GPU), "
            "%d buffers",
            len(loaded), sparse_count, dense_count, len(buffer_weights),
        )
        return loaded

    def _get_device(self) -> torch.device:
        """Return the GPU device for dense computation."""
        return self._gpu_device
