# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Engine components for generative recommendation.

Defines GREngineCoreRequest and GRFeaturesPayload for passing complete
GR feature data through vLLM's EngineCore pipeline (following vllm-omni pattern).

Note: GRInputProcessor, GROutputProcessor, GREngineArgs, GRLLMEngine
are imported directly from their respective modules to avoid circular imports.
"""

from typing import Any
import logging

import msgspec

from vllm.v1.engine import EngineCoreRequest

logger = logging.getLogger("vllm_gr.engine")


class GRFeaturesPayload(msgspec.Struct):
    """
    Serialized GR features payload for transport through EngineCore pipeline.

    Carries ALL features needed by the HSTU model (item_ids, timestamps,
    ratings, weights, candidates, etc.) in a msgspec-serializable format.

    Following the vllm-omni AdditionalInformationPayload pattern.

    Fields:
        uih_keys: UIH feature names, e.g. ["item_id", "action_timestamp", "rating"]
        uih_values_per_key: Per-key int64 value lists (parallel to uih_keys)
        uih_lengths_per_key: Per-key lengths (parallel to uih_keys), each list has batch_size ints
        cand_keys: Candidate feature names, e.g. ["item_movie_id", "item_query_time"]
        cand_values_per_key: Per-key int64 value lists
        cand_lengths_per_key: Per-key lengths
        user_id: User identifier (as string)
        num_recommendations: Number of Top-K recommendations requested
        original_candidates: Original candidate dicts for output mapping
    """

    # UIH (User Interaction History) features
    uih_keys: list[str]
    uih_values_per_key: list[list[int]]
    uih_lengths_per_key: list[list[int]]

    # Candidate features
    cand_keys: list[str]
    cand_values_per_key: list[list[int]]
    cand_lengths_per_key: list[list[int]]

    # Metadata
    user_id: str = ""
    num_recommendations: int = 10
    original_candidates: list[dict[str, Any]] = []


class GREngineCoreRequest(EngineCoreRequest):
    """
    Extended EngineCoreRequest with GR features.

    Follows the vllm-omni OmniEngineCoreRequest pattern: extends the base
    EngineCoreRequest with domain-specific payload fields.

    The gr_features field carries complete GR feature data through the
    EngineCore pipeline to the GRModelRunner.
    """

    # GR feature payload (serializable via msgspec)
    gr_features: GRFeaturesPayload | None = None


__all__ = [
    "GRFeaturesPayload",
    "GREngineCoreRequest",
]
