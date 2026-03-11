# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRInputProcessor — extends vLLM's InputProcessor to handle GRPrompt.

Overrides process_inputs() to detect GRPrompt and convert it into a
GREngineCoreRequest carrying a serialized GRFeaturesPayload. Standard
prompts are delegated to the parent InputProcessor unchanged.
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

from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest

logger = logging.getLogger("vllm_gr.input_processor")


class GRInputProcessor(InputProcessor):
    """Input processor that adds GRPrompt support on top of vLLM's InputProcessor.

    For GRPrompt inputs, extracts user interaction history and candidate
    features, builds a GRFeaturesPayload, and wraps it in a
    GREngineCoreRequest. For all other prompt types, delegates to the
    parent implementation.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike | None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        default_top_k: int = 10,
        uih_feature_names: list[str] | None = None,
        candidate_feature_names: list[str] | None = None,
        primary_id_feature_name: str | None = None,
    ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.default_top_k = default_top_k
        self.uih_feature_names = uih_feature_names or ["item_id"]
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
        if is_gr_prompt(prompt):
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

        return super().process_inputs(
            request_id=request_id,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
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
            params = PoolingParams(task="embed")

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        raw_num_recs = prompt.get("num_recommendations")
        num_recommendations = (
            int(raw_num_recs) if raw_num_recs is not None and raw_num_recs > 0
            else self.default_top_k
        )

        recent_history = user_history[-self.max_seq_len:]
        limited_candidates = candidates[:self.max_candidates]

        uih_keys, uih_values_per_key, uih_lengths_per_key = (
            self._extract_features(recent_history, self.uih_feature_names)
        )
        cand_keys, cand_values_per_key, cand_lengths_per_key = (
            self._extract_features(limited_candidates, self.candidate_feature_names)
        )

        context = prompt.get("context") or {}
        ctx_keys = sorted(context.keys())
        try:
            ctx_values_per_key = [[int(context[k])] for k in ctx_keys]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Context values must be convertible to int. "
                f"Got context={context!r}: {e}"
            ) from e

        gr_features = GRFeaturesPayload(
            uih_keys=uih_keys,
            uih_values_per_key=uih_values_per_key,
            uih_lengths_per_key=uih_lengths_per_key,
            cand_keys=cand_keys,
            cand_values_per_key=cand_values_per_key,
            cand_lengths_per_key=cand_lengths_per_key,
            ctx_keys=ctx_keys,
            ctx_values_per_key=ctx_values_per_key,
            user_id=user_id,
            num_recommendations=num_recommendations,
            original_candidates=[dict(c) for c in limited_candidates],
        )

        pid = self.primary_id_feature_name
        if pid in uih_keys:
            idx = uih_keys.index(pid)
            prompt_token_ids = uih_values_per_key[idx]
        else:
            prompt_token_ids = [0] * len(recent_history) if recent_history else [0]

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
            "cand_keys=%s, cand_len=%d, num_recs=%d",
            request_id, uih_keys, len(recent_history),
            cand_keys, len(limited_candidates), num_recommendations,
        )

        return request

    @staticmethod
    def _extract_features(
        records: list[dict[str, Any]],
        feature_names: list[str],
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """
        Extract multi-key features from a list of records.

        For each feature_name in feature_names, extracts the value from
        each record. Missing values default to 0. Returns flat int lists
        suitable for msgspec serialization.
        """
        active_keys = [key for key in feature_names
                       if any(key in record for record in records)]

        if not active_keys and records:
            if feature_names:
                active_keys = [feature_names[0]]
            else:
                first_record_keys = list(records[0].keys())
                active_keys = [first_record_keys[0]] if first_record_keys else []
            logger.warning(
                "No matching feature keys found in records; "
                "falling back to %s", active_keys,
            )

        values_per_key: list[list[int]] = []
        lengths_per_key: list[int] = []

        for key in active_keys:
            values = [int(record.get(key, 0)) for record in records]
            lengths = len(values)
            values_per_key.append(values)
            lengths_per_key.append(lengths)

        return active_keys, values_per_key, lengths_per_key
