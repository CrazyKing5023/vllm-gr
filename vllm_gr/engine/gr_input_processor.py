# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRInputProcessor — extends vLLM's InputProcessor to handle GRPrompt.

Overrides process_inputs() to require GRPrompt inputs and convert them into a
GREngineCoreRequest carrying a serialized GRFeaturesPayload.
"""

from typing import Any
from collections.abc import Mapping
import time
import logging

from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine import EngineCoreRequest
from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

from vllm_gr.gr_schema import (
    attach_schema_fingerprint as _attach_schema_fingerprint,
    build_schema_fingerprint as _build_schema_fingerprint,
    extract_context_features as _extract_context_features_helper,
    extract_features as _extract_features_helper,
    validate_context_group as _validate_context_group,
    validate_feature_group as _validate_feature_group,
)
from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest

logger = logging.getLogger("vllm_gr.input_processor")


class GRInputProcessor(InputProcessor):
    """Input processor for the GR-only engine contract.

    For GRPrompt inputs, extracts user interaction history and candidate
    features, builds a GRFeaturesPayload, and wraps it in a
    GREngineCoreRequest.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike | None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        uih_feature_names: list[str] | None = None,
        candidate_feature_names: list[str] | None = None,
        primary_id_feature_name: str | None = None,
        full_uih_feature_names: list[str] | None = None,
        context_feature_names: list[str] | None = None,
    ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.full_uih_feature_names = list(
            full_uih_feature_names or uih_feature_names or ["item_id"]
        )
        self.context_feature_names = list(context_feature_names or [])
        unexpected_context_features = [
            key for key in self.context_feature_names
            if key not in self.full_uih_feature_names
        ]
        if unexpected_context_features:
            raise ValueError(
                "Context feature names must be a subset of full UIH schema; "
                f"unexpected {unexpected_context_features!r}"
            )
        context_feature_name_set = set(self.context_feature_names)
        self.uih_feature_names = [
            key for key in self.full_uih_feature_names
            if key not in context_feature_name_set
        ]
        self.candidate_feature_names = (
            candidate_feature_names or ["item_candidate_id"]
        )
        self.primary_id_feature_name = primary_id_feature_name or "item_id"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> EngineCoreRequest:
        if not is_gr_prompt(prompt):
            raise TypeError(
                "GRInputProcessor only supports GRPrompt inputs; "
                f"got {type(prompt).__name__}"
            )

        return self._process_gr_prompt(
            request_id=request_id,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

    def _process_gr_prompt(
        self,
        request_id: str,
        prompt: GRPrompt,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> GREngineCoreRequest:
        """
        Process a GRPrompt into a GREngineCoreRequest with full feature payload.

        Reuses parent validation where applicable, then builds GR-specific
        request with GRFeaturesPayload.
        """
        self._validate_lora(lora_request)

        if arrival_time is None:
            arrival_time = time.time()

        if not isinstance(params, PoolingParams):
            raise TypeError(
                "GRInputProcessor only accepts PoolingParams for GR prompts; "
                f"got {type(params).__name__}"
            )

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        recent_history = user_history[-self.max_seq_len:]

        uih_keys, uih_values_per_key, uih_lengths_per_key = (
            self._extract_features(
                recent_history,
                self.uih_feature_names,
                group_name="uih",
            )
        )
        cand_keys, cand_values_per_key, cand_lengths_per_key = (
            self._extract_features(
                candidates,
                self.candidate_feature_names,
                group_name="cand",
            )
        )

        context = prompt.get("context") or {}
        ctx_keys, ctx_values_per_key = self._extract_context_features(
            context,
            feature_names=self.context_feature_names,
        )

        (
            uih_keys,
            cand_keys,
            ctx_keys,
            schema_fingerprint,
        ) = self.attach_schema_fingerprint(
            uih_keys=uih_keys,
            cand_keys=cand_keys,
            ctx_keys=ctx_keys,
        )
        self.validate_feature_group(
            "uih",
            uih_keys,
            uih_values_per_key,
            uih_lengths_per_key,
        )
        self.validate_feature_group(
            "cand",
            cand_keys,
            cand_values_per_key,
            cand_lengths_per_key,
        )
        self.validate_context_group(ctx_keys, ctx_values_per_key)

        gr_features = GRFeaturesPayload(
            uih_keys=uih_keys,
            uih_values_per_key=uih_values_per_key,
            uih_lengths_per_key=uih_lengths_per_key,
            cand_keys=cand_keys,
            cand_values_per_key=cand_values_per_key,
            cand_lengths_per_key=cand_lengths_per_key,
            ctx_keys=ctx_keys,
            ctx_values_per_key=ctx_values_per_key,
            full_uih_keys=list(self.full_uih_feature_names),
            user_id=user_id,
            original_candidates=[dict(c) for c in candidates],
        )

        pid = self.primary_id_feature_name
        if pid in uih_keys:
            idx = uih_keys.index(pid)
            prompt_token_ids = uih_values_per_key[idx]
        elif recent_history:
            raise ValueError(
                f"UIH features are missing primary id feature {pid!r}; "
                "request schema does not match model configuration"
            )
        else:
            prompt_token_ids = [0]

        request = GREngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=None,
            pooling_params=params,
            eos_token_id=None,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=None,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            trace_headers=trace_headers,
            gr_features=gr_features,
        )

        logger.debug(
            "Processed GR request %s: uih_keys=%s, uih_len=%d, "
            "cand_keys=%s, cand_len=%d, schema=%s",
            request_id, uih_keys, len(recent_history),
            cand_keys, len(candidates),
            schema_fingerprint,
        )

        return request

    @staticmethod
    def attach_schema_fingerprint(
        *,
        uih_keys: list[str],
        cand_keys: list[str],
        ctx_keys: list[str],
    ) -> tuple[list[str], list[str], list[str], str]:
        """Attach a stable request-schema fingerprint for logging/debugging."""
        return _attach_schema_fingerprint(
            uih_keys=uih_keys,
            cand_keys=cand_keys,
            ctx_keys=ctx_keys,
        )

    @staticmethod
    def build_schema_fingerprint(
        *,
        uih_keys: list[str],
        cand_keys: list[str],
        ctx_keys: list[str],
    ) -> str:
        """Return a stable local fingerprint for a request schema."""
        return _build_schema_fingerprint(
            uih_keys=uih_keys,
            cand_keys=cand_keys,
            ctx_keys=ctx_keys,
        )

    @staticmethod
    def validate_feature_group(
        group_name: str,
        keys: list[str],
        values_per_key: list[list[int]],
        lengths_per_key: list[int],
    ) -> None:
        """Ensure a serialized feature group is internally consistent."""
        _validate_feature_group(
            group_name=group_name,
            keys=keys,
            values_per_key=values_per_key,
            lengths_per_key=lengths_per_key,
        )

    @staticmethod
    def validate_context_group(
        ctx_keys: list[str],
        ctx_values_per_key: list[list[int]],
    ) -> None:
        """Context features are per-request scalars and must stay 1:1."""
        _validate_context_group(
            ctx_keys=ctx_keys,
            ctx_values_per_key=ctx_values_per_key,
        )

    @staticmethod
    def _extract_context_features(
        context: Mapping[str, Any],
        feature_names: list[str] | None = None,
    ) -> tuple[list[str], list[list[int]]]:
        return _extract_context_features_helper(
            context,
            feature_names=feature_names,
        )

    @staticmethod
    def _extract_features(
        records: list[dict[str, Any]],
        feature_names: list[str],
        group_name: str = "features",
    ) -> tuple[list[str], list[list[int]], list[int]]:
        return _extract_features_helper(
            records=records,
            feature_names=feature_names,
            group_name=group_name,
        )
