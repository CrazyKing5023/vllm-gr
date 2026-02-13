# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRInputProcessor: Input processor for generative recommendation.

Converts GRPrompt into a GREngineCoreRequest with serialized
GRFeaturesPayload. This ensures ALL features (item_ids, timestamps,
ratings, action_types, etc.) are properly serialized and transmitted
through vLLM's EngineCore pipeline to the GRModelRunner.

Replaces the old GRProcessor which only extracted item_ids.
"""

from typing import Any, Optional, List, Dict
import time
import logging

from vllm_gr.inputs.data import GRPrompt
from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest

logger = logging.getLogger("vllm_gr.input_processor")


class GRInputProcessor:
    """
    GR Input Processor — converts GRPrompt to GREngineCoreRequest.

    This processor:
    1. Extracts ALL features from GRPrompt (item_id, timestamp, rating,
       action_type, category_id, etc.) — not just item_id.
    2. Serializes them into GRFeaturesPayload (msgspec-compatible).
    3. Creates a GREngineCoreRequest with both:
       - prompt_token_ids (item_ids, for vLLM compatibility)
       - gr_features (full feature payload, for GRModelRunner)

    Data format correspondence with generative-recommenders:
    - GRPrompt.user_history → uih_features (multi-key KJT)
    - GRPrompt.candidates → candidate_features (multi-key KJT)
    """

    # Default feature keys to extract from user history
    DEFAULT_UIH_KEYS = ["item_id", "timestamp", "rating", "action_type"]
    # Default feature keys to extract from candidates
    DEFAULT_CAND_KEYS = ["item_id", "category_id"]

    def __init__(
        self,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        uih_feature_names: Optional[List[str]] = None,
        candidate_feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            max_seq_len: Maximum user history sequence length
            max_candidates: Maximum number of candidate items
            uih_feature_names: Feature names to extract from user_history
                (default: ["item_id", "timestamp", "rating", "action_type"])
            candidate_feature_names: Feature names to extract from candidates
                (default: ["item_id", "category_id"])
        """
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.uih_feature_names = uih_feature_names or self.DEFAULT_UIH_KEYS
        self.candidate_feature_names = (
            candidate_feature_names or self.DEFAULT_CAND_KEYS
        )

    def process_gr_prompt(
        self,
        request_id: str,
        prompt: GRPrompt,
        num_recommendations: int = 10,
        arrival_time: Optional[float] = None,
        priority: int = 0,
    ) -> GREngineCoreRequest:
        """
        Process GRPrompt into GREngineCoreRequest.

        This is the main entry point, called by GRLLMEngine.add_gr_request().

        Args:
            request_id: Unique request ID
            prompt: User request in GRPrompt format
            num_recommendations: Number of Top-K recommendations
            arrival_time: Request arrival time
            priority: Request priority

        Returns:
            GREngineCoreRequest with full feature payload
        """
        if arrival_time is None:
            arrival_time = time.time()

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])

        # Truncate to max lengths
        recent_history = user_history[-self.max_seq_len:]
        limited_candidates = candidates[:self.max_candidates]

        # ---------------------------------------------------------------
        # 1. Extract UIH features (multi-key)
        # ---------------------------------------------------------------
        uih_keys, uih_values_per_key, uih_lengths_per_key = (
            self._extract_features(
                records=recent_history,
                feature_names=self.uih_feature_names,
            )
        )

        # ---------------------------------------------------------------
        # 2. Extract Candidate features (multi-key)
        # ---------------------------------------------------------------
        cand_keys, cand_values_per_key, cand_lengths_per_key = (
            self._extract_features(
                records=limited_candidates,
                feature_names=self.candidate_feature_names,
            )
        )

        # ---------------------------------------------------------------
        # 3. Build GRFeaturesPayload
        # ---------------------------------------------------------------
        gr_features = GRFeaturesPayload(
            uih_keys=uih_keys,
            uih_values_per_key=uih_values_per_key,
            uih_lengths_per_key=uih_lengths_per_key,
            cand_keys=cand_keys,
            cand_values_per_key=cand_values_per_key,
            cand_lengths_per_key=cand_lengths_per_key,
            user_id=user_id,
            num_recommendations=num_recommendations,
            # Keep original candidate dicts for output mapping
            original_candidates=[dict(c) for c in limited_candidates],
        )

        # ---------------------------------------------------------------
        # 4. Build prompt_token_ids for vLLM compatibility
        # ---------------------------------------------------------------
        # Use item_ids from UIH as "token IDs" so vLLM's length/sequence
        # logic works correctly. If item_id is in uih_keys, use those
        # values; otherwise use a dummy [0].
        if "item_id" in uih_keys:
            idx = uih_keys.index("item_id")
            prompt_token_ids = uih_values_per_key[idx]
        else:
            prompt_token_ids = [0] * len(recent_history) if recent_history else [0]

        # ---------------------------------------------------------------
        # 5. Create GREngineCoreRequest
        # ---------------------------------------------------------------
        from vllm.pooling_params import PoolingParams

        request = GREngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            eos_token_id=None,
            arrival_time=arrival_time,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            # Fields with defaults
            priority=priority,
            # GR-specific payload
            gr_features=gr_features,
        )

        logger.debug(
            f"Processed GR request {request_id}: "
            f"uih_keys={uih_keys}, uih_len={len(recent_history)}, "
            f"cand_keys={cand_keys}, cand_len={len(limited_candidates)}, "
            f"num_recs={num_recommendations}"
        )

        return request

    def _extract_features(
        self,
        records: List[Dict[str, Any]],
        feature_names: List[str],
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        """
        Extract multi-key features from a list of records.

        For each feature_name in feature_names, extracts the value from
        each record. Missing values default to 0. Each key produces a
        flat list of int64 values and a length list (one entry per record,
        all 1 for dense features — extensible to variable-length later).

        Args:
            records: List of dicts (ItemInteraction or CandidateItem)
            feature_names: Feature keys to extract

        Returns:
            (keys, values_per_key, lengths_per_key)
            - keys: list of actual feature names found
            - values_per_key: list of int lists (parallel to keys)
            - lengths_per_key: list of int lists (parallel to keys)
        """
        # Only include keys that appear in at least one record
        active_keys = []
        for key in feature_names:
            if any(key in record for record in records):
                active_keys.append(key)

        # If no active keys but records exist, at least include "item_id"
        if not active_keys and records:
            active_keys = ["item_id"]

        values_per_key: list[list[int]] = []
        lengths_per_key: list[list[int]] = []

        for key in active_keys:
            values = [int(record.get(key, 0)) for record in records]
            # Dense feature: length = 1 per record
            lengths = [1] * len(records)
            values_per_key.append(values)
            lengths_per_key.append(lengths)

        return active_keys, values_per_key, lengths_per_key
