# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR — restored to match original generative-recommenders interface.
# Licensed under the Apache License, Version 2.0

"""
DlrmHSTU Main Model Class

Generative recommendation model based on HSTU (Hierarchical Sequential Transducer Unit).
Supports multi-task prediction and user-item interaction modeling.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from torch.autograd.profiler import record_function
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

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
from vllm_gr.utils import (
    ReliableOps,
    get_hstu_autocast_context,
)


logger: logging.Logger = logging.getLogger(__name__)


def _cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Compute inclusive prefix-sum with leading zero (complete cumsum)."""
    return ReliableOps.asynchronous_complete_cumsum(lengths)


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
    hstu_transducer_embedding_dim: int = 0
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
    DLRM_v3 HSTU Model.

    Restored to match original generative-recommenders interface:
    - __init__ accepts embedding_tables and creates EmbeddingCollection
    - preprocess() merges KJTs and performs embedding lookup
    - forward() accepts KeyedJaggedTensor inputs
    """

    def __init__(
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool = True,
        is_dense: bool = False,
        bf16_training: bool = True,
        embedding_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")

        self._hstu_configs = hstu_configs
        self._bf16_training: bool = bf16_training
        self._embedding_device = self._consume_embedding_device(embedding_device)
        set_static_max_seq_lens([self._hstu_configs.max_seq_len])

        if not is_dense:
            # Phase 1 keeps embedding execution on the conservative runtime device.
            self._embedding_collection: EmbeddingCollection = EmbeddingCollection(
                tables=list(embedding_tables.values()),
                need_indices=False,
                device=self._embedding_device,
            )

        # multitask configs must be sorted by task types
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
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

        self._additional_embedding_features: List[str] = [
            uih_feature_name
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self._hstu_configs.merge_uih_candidate_feature_mapping
            if (
                candidate_feature_name
                in self._hstu_configs.item_embedding_feature_names
            )
            and (uih_feature_name in self._hstu_configs.user_embedding_feature_names)
            and (uih_feature_name != self._hstu_configs.uih_post_id_feature_name)
        ]

        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            hidden_dim=hstu_configs.hstu_preprocessor_hidden_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            action_embedding_dim=8,
            action_feature_name=self._hstu_configs.uih_weight_feature_name,
            action_weights=self._hstu_configs.action_weights,
            additional_embedding_features=self._additional_embedding_features,
            action_embedding_init_std=self._hstu_configs.action_embedding_init_std,
            is_inference=is_inference,
        )

        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_seq_len=sum(
                dict(hstu_configs.contextual_feature_to_max_length).values()
            ),
            is_inference=self._is_inference,
        )

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
                        (60 * 60, 24),
                        (24 * 60 * 60, 7),
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

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

        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim
                * len(self._hstu_configs.item_embedding_feature_names),
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    @staticmethod
    def _consume_embedding_device(
        embedding_device: Optional[torch.device],
    ) -> torch.device:
        """Consume an already-resolved embedding device from runtime policy.

        DlrmHSTU should not infer a platform policy on its own. The caller is
        expected to pass the resolved phase-1 device explicitly. A CPU fallback
        is kept only as a compatibility path for older call-sites.
        """
        if embedding_device is None:
            logger.warning(
                "DlrmHSTU received no explicit embedding_device; "
                "falling back to CPU for compatibility. "
                "Callers should pass the resolved runtime policy result."
            )
            return torch.device("cpu")
        return torch.device(embedding_device)

    # =========================================================================
    # preprocess: KJT merge → EmbeddingCollection → payload + seq_embeddings
    # =========================================================================

    def preprocess(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        """Merge KJTs, run EmbeddingCollection, extract payload and seq_embeddings."""
        merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=uih_features.keys() + candidates_features.keys(),
            values=torch.cat(
                [uih_features.values(), candidates_features.values()], dim=0,
            ),
            lengths=torch.cat(
                [uih_features.lengths(), candidates_features.lengths()], dim=0,
            ),
        )
        seq_embeddings_dict = self._embedding_collection(merged_sparse_features)

        num_candidates = fx_mark_length_features(
            candidates_features.lengths().view(
                len(candidates_features.keys()), -1)
        )[0]
        max_num_candidates = fx_infer_max_len(num_candidates)
        uih_seq_lengths = uih_features[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths()
        max_uih_len = fx_infer_max_len(uih_seq_lengths)

        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                candidate_feature_name
                not in self._hstu_configs.item_embedding_feature_names
                and uih_feature_name
                not in self._hstu_configs.user_embedding_feature_names
            ):
                values_left = uih_features[uih_feature_name].values()
                if self._is_inference and (
                    candidate_feature_name
                    == self._hstu_configs.candidates_weight_feature_name
                    or candidate_feature_name
                    == self._hstu_configs.candidates_watchtime_feature_name
                ):
                    total_candidates = torch.sum(num_candidates).item()
                    values_right = torch.zeros(
                        total_candidates,  # pyre-ignore
                        dtype=torch.int64,
                        device=values_left.device,
                    )
                else:
                    values_right = candidates_features[candidate_feature_name].values()
                payload_features[uih_feature_name] = values_left
                payload_features[candidate_feature_name] = values_right

        payload_features["uih_offsets"] = _cumsum(uih_seq_lengths)
        payload_features["candidate_offsets"] = _cumsum(num_candidates)

        seq_embeddings = {
            k: SequenceEmbedding(
                lengths=seq_embeddings_dict[k].lengths(),
                embedding=seq_embeddings_dict[k].values(),
            )
            for k in self._hstu_configs.user_embedding_feature_names
            + self._hstu_configs.item_embedding_feature_names
        }

        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    # =========================================================================
    # _construct_payload: add contextual + additional embeddings to payload
    # =========================================================================

    def _construct_payload(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> Dict[str, torch.Tensor]:
        if len(self._hstu_configs.contextual_feature_to_max_length) > 0:
            contextual_offsets: List[torch.Tensor] = []
            for x in self._hstu_configs.contextual_feature_to_max_length.keys():
                contextual_offsets.append(
                    _cumsum(seq_embeddings[x].lengths)
                )
        else:
            contextual_offsets = torch.empty((0, 0))  # type: ignore[assignment]
        return {
            **payload_features,
            **{
                x: seq_embeddings[x].embedding
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            },
            **{
                x + "_offsets": contextual_offsets[i]
                for i, x in enumerate(
                    list(self._hstu_configs.contextual_feature_to_max_length.keys())
                )
            },
            **{
                x: seq_embeddings[x].embedding
                for x in self._additional_embedding_features
            },
        }

    # =========================================================================
    # _item_forward / _user_forward
    # =========================================================================

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:
        all_embeddings = torch.cat(
            [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.item_embedding_feature_names
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
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths

        source_timestamps = concat_2D_jagged(
            max_seq_len=max_uih_len + max_candidates,
            max_len_left=max_uih_len,
            offsets_left=payload_features["uih_offsets"],
            values_left=payload_features[
                self._hstu_configs.uih_action_time_feature_name
            ].unsqueeze(-1),
            max_len_right=max_candidates,
            offsets_right=payload_features["candidate_offsets"],
            values_right=payload_features[
                self._hstu_configs.candidates_querytime_feature_name
            ].unsqueeze(-1),
            kernel=self.hammer_kernel(),
        ).squeeze(-1)

        total_targets = int(num_candidates.sum().item())
        embedding = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].embedding
        dtype = embedding.dtype

        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)

        with get_hstu_autocast_context(
            embedding,
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
                seq_payloads=self._construct_payload(
                    payload_features=payload_features,
                    seq_embeddings=seq_embeddings,
                ),
                num_targets=num_candidates,
            )

        return candidates_user_embeddings.to(dtype)

    # =========================================================================
    # main_forward: merge UIH+candidate embeddings, then item/user/multitask
    # =========================================================================

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
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if uih_feature_name in seq_embeddings:
                seq_embeddings[uih_feature_name] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_seq_len=max_uih_len + max_num_candidates,
                        max_len_left=max_uih_len,
                        offsets_left=_cumsum(uih_seq_lengths),
                        values_left=seq_embeddings[uih_feature_name].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=_cumsum(num_candidates),
                        values_right=seq_embeddings[
                            candidate_feature_name
                        ].embedding,
                        kernel=self.hammer_kernel(),
                    ),
                )

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
            if self._is_inference:
                supervision_labels: Dict[str, torch.Tensor] = {}
                supervision_weights: Dict[str, torch.Tensor] = {}
            else:
                supervision_labels, supervision_weights = (
                    _get_supervision_labels_and_weights(
                        supervision_bitmasks=payload_features[
                            self._hstu_configs.candidates_weight_feature_name
                        ],
                        watchtime_sequence=payload_features[
                            self._hstu_configs.candidates_watchtime_feature_name
                        ],
                        task_configs=self._multitask_configs,
                    )
                )
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    supervision_labels=supervision_labels,
                    supervision_weights=supervision_weights,
                )
            )

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and self.training:
            for i, task in enumerate(self._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    # =========================================================================
    # forward: KJT → preprocess → main_forward
    # =========================================================================

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        with record_function("## preprocess ##"):
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )

        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
