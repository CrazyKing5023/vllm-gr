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

    DEFAULT_UIH_KEYS = ["item_id", "timestamp", "rating", "action_type"]
    DEFAULT_CAND_KEYS = ["item_id", "category_id"]

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike | None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        uih_feature_names: list[str] | None = None,
        candidate_feature_names: list[str] | None = None,
    ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.uih_feature_names = uih_feature_names or self.DEFAULT_UIH_KEYS
        self.candidate_feature_names = (
            candidate_feature_names or self.DEFAULT_CAND_KEYS
        )

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

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        num_recommendations = prompt.get("num_recommendations", 10)

        recent_history = user_history[-self.max_seq_len:]
        limited_candidates = candidates[:self.max_candidates]

        uih_keys, uih_values_per_key, uih_lengths_per_key = (
            self._extract_features(recent_history, self.uih_feature_names)
        )
        cand_keys, cand_values_per_key, cand_lengths_per_key = (
            self._extract_features(limited_candidates, self.candidate_feature_names)
        )

        gr_features = GRFeaturesPayload(
            uih_keys=uih_keys,
            uih_values_per_key=uih_values_per_key,
            uih_lengths_per_key=uih_lengths_per_key,
            cand_keys=cand_keys,
            cand_values_per_key=cand_values_per_key,
            cand_lengths_per_key=cand_lengths_per_key,
            user_id=user_id,
            num_recommendations=num_recommendations,
            original_candidates=[dict(c) for c in limited_candidates],
        )

        if "item_id" in uih_keys:
            idx = uih_keys.index("item_id")
            prompt_token_ids = uih_values_per_key[idx]
        else:
            prompt_token_ids = [0] * len(recent_history) if recent_history else [0]

        pooling_params = PoolingParams(task="embed")

        request = GREngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=None,
            pooling_params=pooling_params,
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
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        """
        Extract multi-key features from a list of records.

        For each feature_name in feature_names, extracts the value from
        each record. Missing values default to 0. Returns flat int lists
        suitable for msgspec serialization.
        """
        active_keys = [key for key in feature_names
                       if any(key in record for record in records)]

        if not active_keys and records:
            active_keys = ["item_id"]

        values_per_key: list[list[int]] = []
        lengths_per_key: list[list[int]] = []

        for key in active_keys:
            values = [int(record.get(key, 0)) for record in records]
            lengths = [1] * len(records)
            values_per_key.append(values)
            lengths_per_key.append(lengths)

        return active_keys, values_per_key, lengths_per_key
