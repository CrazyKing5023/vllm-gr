# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0

"""
DlrmHSTU 主模型类

基于 HSTU (Hierarchical Sequential Transducer Unit) 架构的生成式推荐模型。
支持多任务预测和用户-物品交互建模。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from torch.autograd.profiler import record_function

from vllm_gr.models.dlrm_v3.common import (
    HammerModule,
    fx_infer_max_len,
    fx_mark_length_features,
    init_mlp_weights_optional_bias,
    set_static_max_seq_lens,
)
from vllm_gr.models.dlrm_v3.modules.hstu_transducer import HSTUTransducer
from vllm_gr.models.dlrm_v3.modules.multitask_module import (
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)
from vllm_gr.models.dlrm_v3.modules.positional_encoder import HSTUPositionalEncoder
from vllm_gr.models.dlrm_v3.modules.postprocessors import (
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from vllm_gr.models.dlrm_v3.modules.preprocessors import ContextualPreprocessor
from vllm_gr.models.dlrm_v3.modules.stu import STU, STULayer, STULayerConfig, STUStack
from vllm_gr.ops.jagged_tensors import concat_2D_jagged
from vllm_gr.ops.layer_norm import LayerNorm, SwishLayerNorm


logger: logging.Logger = logging.getLogger(__name__)


def complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Compute complete cumulative sum (starting with 0)."""
    try:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except (RuntimeError, AttributeError):
        cumsum = torch.zeros(lengths.size(0) + 1, dtype=lengths.dtype, device=lengths.device)
        cumsum[1:] = torch.cumsum(lengths, dim=0)
        return cumsum


class SequenceEmbedding(NamedTuple):
    """Container for sequence embeddings with lengths."""
    lengths: torch.Tensor
    embedding: torch.Tensor


@dataclass
class DlrmHSTUConfig:
    """Configuration for DLRM HSTU model."""
    max_seq_len: int = 16384
    max_num_candidates: int = 10
    max_num_candidates_inference: int = 5
    hstu_num_heads: int = 1
    hstu_attn_linear_dim: int = 256
    hstu_attn_qk_dim: int = 128
    hstu_attn_num_layers: int = 12
    hstu_embedding_table_dim: int = 192
    hstu_preprocessor_hidden_dim: int = 256
    hstu_transducer_embedding_dim: int = 512
    hstu_group_norm: bool = False
    hstu_input_dropout_ratio: float = 0.2
    hstu_linear_dropout_rate: float = 0.2
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    candidates_weight_feature_name: str = ""
    candidates_watchtime_feature_name: str = ""
    candidates_querytime_feature_name: str = ""
    causal_multitask_weights: float = 0.2
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    user_embedding_feature_names: List[str] = field(default_factory=list)
    item_embedding_feature_names: List[str] = field(default_factory=list)
    uih_post_id_feature_name: str = ""
    uih_action_time_feature_name: str = ""
    uih_weight_feature_name: str = ""
    hstu_uih_feature_names: List[str] = field(default_factory=list)
    hstu_candidate_feature_names: List[str] = field(default_factory=list)
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(default_factory=list)
    action_weights: Optional[List[int]] = None
    action_embedding_init_std: float = 0.1
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False


def _get_supervision_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    watchtime_sequence: torch.Tensor,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Extract supervision labels and weights for each task."""
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            supervision_labels[task.task_name] = watchtime_sequence.to(torch.float32)
        elif task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
            supervision_labels[task.task_name] = (
                torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
            ).to(torch.float32)
        else:
            raise RuntimeError("Unsupported MultitaskTaskType")
    
    return supervision_labels, supervision_weights


class DlrmHSTU(HammerModule):
    """
    DLRM_v3 HSTU 模型
    
    基于 Hierarchical Sequential Transducer Unit 的生成式推荐模型。
    支持稀疏-稠密分离、因果注意力、目标感知注意力和多任务学习。
    """
    
    def __init__(
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_collection: Optional[torch.nn.Module] = None,
        is_inference: bool = True,
        is_dense: bool = False,
        bf16_training: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")
        
        self._hstu_configs = hstu_configs
        self._bf16_training: bool = bf16_training
        set_static_max_seq_lens([self._hstu_configs.max_seq_len])

        # Embedding collection (optional, can be managed externally)
        self._embedding_collection = embedding_collection
        self._is_dense = is_dense

        # Multitask module
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
        if self._multitask_configs:
            self._multitask_module = DefaultMultitaskModule(
                task_configs=self._multitask_configs,
                embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                    torch.nn.Linear(in_features=in_dim, out_features=512),
                    SwishLayerNorm(512),
                    torch.nn.Linear(in_features=512, out_features=num_tasks),
                ).apply(init_mlp_weights_optional_bias),
                causal_multitask_weights=hstu_configs.causal_multitask_weights,
                is_inference=self._is_inference,
            )
        else:
            self._multitask_module = None

        # Preprocessor
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            hidden_dim=hstu_configs.hstu_preprocessor_hidden_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            action_embedding_dim=8,
            action_feature_name=self._hstu_configs.uih_weight_feature_name,
            action_weights=self._hstu_configs.action_weights,
            action_embedding_init_std=self._hstu_configs.action_embedding_init_std,
            is_inference=is_inference,
        )

        # Positional encoder
        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_seq_len=sum(dict(hstu_configs.contextual_feature_to_max_length).values()),
            is_inference=self._is_inference,
        )

        # Postprocessor
        if hstu_configs.enable_postprocessor:
            if hstu_configs.use_layer_norm_postprocessor:
                postprocessor = LayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
            else:
                postprocessor = TimestampLayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    time_duration_features=[
                        (60 * 60, 24),  # hour of day
                        (24 * 60 * 60, 7),  # day of week
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

        # Construct HSTU (STU stack)
        stu_module: STU = STUStack(
            stu_list=[
                STULayer(
                    config=STULayerConfig(
                        embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                        num_heads=hstu_configs.hstu_num_heads,
                        hidden_dim=hstu_configs.hstu_attn_linear_dim,
                        attention_dim=hstu_configs.hstu_attn_qk_dim,
                        output_dropout_ratio=hstu_configs.hstu_linear_dropout_rate,
                        use_group_norm=hstu_configs.hstu_group_norm,
                        causal=True,
                        target_aware=True,
                        max_attn_len=None,
                        attn_alpha=None,
                        recompute_normed_x=True,
                        recompute_uvqk=True,
                        recompute_y=True,
                        sort_by_length=True,
                        contextual_seq_len=0,
                    ),
                    is_inference=is_inference,
                )
                for _ in range(hstu_configs.hstu_attn_num_layers)
            ],
            is_inference=is_inference,
        )

        # HSTU Transducer
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            stu_module=stu_module,
            input_preprocessor=preprocessor,
            output_postprocessor=postprocessor,
            input_dropout_ratio=hstu_configs.hstu_input_dropout_ratio,
            positional_encoder=positional_encoder,
            is_inference=self._is_inference,
            return_full_embeddings=False,
            listwise=False,
        )

        # Item embedding MLP
        num_item_features = len(self._hstu_configs.item_embedding_feature_names) or 1
        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim * num_item_features,
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:
        """Compute item embeddings from features."""
        if not self._hstu_configs.item_embedding_feature_names:
            # Fallback: use first available embedding
            first_key = next(iter(seq_embeddings.keys()))
            return self._item_embedding_mlp(seq_embeddings[first_key].embedding)
        
        all_embeddings = torch.cat(
            [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.item_embedding_feature_names
                if name in seq_embeddings
            ],
            dim=-1,
        )
        return self._item_embedding_mlp(all_embeddings)

    def _user_forward(
        self,
        max_uih_len: int,
        max_candidates: int,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute user embeddings using HSTU transducer."""
        source_key = self._hstu_configs.uih_post_id_feature_name
        if not source_key or source_key not in seq_embeddings:
            source_key = next(iter(seq_embeddings.keys()))
        
        source_lengths = seq_embeddings[source_key].lengths
        embedding = seq_embeddings[source_key].embedding
        total_targets = int(num_candidates.sum().item())
        
        # 创建时间戳（简化实现）
        total_len = embedding.size(0)
        source_timestamps = torch.zeros(total_len, device=embedding.device, dtype=torch.float32)
        
        dtype = embedding.dtype
        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference) and self._bf16_training,
        ):
            candidates_user_embeddings, _ = self._hstu_transducer(
                max_uih_len=max_uih_len,
                max_targets=max_candidates,
                total_uih_len=source_timestamps.numel() - total_targets,
                total_targets=total_targets,
                seq_embeddings=embedding,
                seq_lengths=source_lengths,
                seq_timestamps=source_timestamps,
                seq_payloads=payload_features,
                num_targets=num_candidates,
            )
        
        return candidates_user_embeddings.to(dtype)

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Main forward computation."""
        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(seq_embeddings)
        
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                num_candidates=num_candidates,
            )
        
        with record_function("## multitask_module ##"):
            if self._multitask_module is not None:
                mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                    self._multitask_module(
                        encoded_user_embeddings=candidates_user_embeddings,
                        item_embeddings=candidates_item_embeddings,
                        supervision_labels={},
                        supervision_weights={},
                    )
                )
            else:
                # Simple dot product scoring
                mt_target_preds = (candidates_user_embeddings * candidates_item_embeddings).sum(dim=-1, keepdim=True)
                mt_target_preds = mt_target_preds.t()  # (1, L)
                mt_target_labels = None
                mt_target_weights = None
                mt_losses = None

        aux_losses: Dict[str, torch.Tensor] = {}

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    def forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        num_candidates: torch.Tensor,
        max_uih_len: Optional[int] = None,
        max_num_candidates: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass of DLRM HSTU model.
        
        Args:
            seq_embeddings: Dictionary of sequence embeddings
            num_candidates: Number of candidates per batch element
            max_uih_len: Maximum UIH length (computed if not provided)
            max_num_candidates: Maximum candidates (computed if not provided)
        
        Returns:
            (user_embeddings, item_embeddings, aux_losses, predictions, labels, weights)
        """
        # Get first sequence to compute lengths
        first_key = next(iter(seq_embeddings.keys()))
        uih_seq_lengths = seq_embeddings[first_key].lengths
        
        if max_uih_len is None:
            max_uih_len = fx_infer_max_len(uih_seq_lengths)
        if max_num_candidates is None:
            max_num_candidates = fx_infer_max_len(num_candidates)
        
        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features={},
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )

