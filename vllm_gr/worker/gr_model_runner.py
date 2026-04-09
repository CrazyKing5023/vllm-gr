# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""GPU GR model runner."""

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin


class GRModelRunner(GRFeatureRunnerMixin, GPUModelRunner):
    """GPU model runner that injects GR features into HSTU inputs."""

    def _build_attention_metadata(self, *args, **kwargs):
        self._prepare_hstu_attention_builders()
        return super()._build_attention_metadata(*args, **kwargs)

    @staticmethod
    def _extract_feature_length(lengths: torch.Tensor | None) -> int:
        if lengths is None or lengths.numel() == 0:
            return 0
        return int(lengths.max().item())

    def _build_hstu_request_lengths(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if not self._gr_features_cache:
            return None

        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config is None or not bool(getattr(hf_config, "enable_kv_cache", False)):
            return None

        batch_req_ids = list(self.input_batch.req_ids)
        if not batch_req_ids:
            return None

        num_targets: list[int] = []
        history_lengths: list[int] = []
        for req_id in batch_req_ids:
            features = self._gr_features_cache.get(req_id)
            if features is None:
                return None
            num_targets.append(
                self._extract_feature_length(features.get("cand_lengths_per_key"))
            )
            history_lengths.append(
                self._extract_feature_length(features.get("uih_lengths_per_key"))
            )

        contextual_seq_len = int(
            sum(getattr(hf_config, "contextual_feature_to_max_length", {}).values())
        )
        return (
            torch.tensor(num_targets, dtype=torch.int32, device=self.device),
            torch.tensor(history_lengths, dtype=torch.int32, device=self.device),
            contextual_seq_len,
        )

    def _prepare_hstu_attention_builders(self) -> None:
        from vllm_gr.attention.backends.hstu_attn import (
            HSTUAttentionMetadataBuilder,
        )

        request_lengths = self._build_hstu_request_lengths()
        for kv_cache_groups in self.attn_groups:
            for attn_group in kv_cache_groups:
                for builder in attn_group.metadata_builders:
                    if not isinstance(builder, HSTUAttentionMetadataBuilder):
                        continue
                    if request_lengths is None:
                        builder.clear_request_lengths()
                    else:
                        num_targets, history_lengths, contextual_seq_len = request_lengths
                        builder.set_request_lengths(
                            num_targets=num_targets,
                            history_lengths=history_lengths,
                            contextual_seq_len=contextual_seq_len,
                        )
