# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Engine components for generative recommendation.

Defines GREngineCoreRequest and GRFeaturesPayload for passing complete
GR feature data through vLLM's EngineCore pipeline (following vllm-omni pattern).

Note: GRInputProcessor, GROutputProcessor, GREngineArgs, GRLLMEngine
are imported directly from their respective modules to avoid circular imports.

msgspec serialization note:
    EngineCoreRequest uses array_like=True, so fields are serialized
    positionally. GREngineCoreRequest appends gr_features at the end.
    This works transparently for InprocClient (no serialization).
    For multi-process mode (ZMQ), the monkey patch must replace
    EngineCoreRequest in the engine process BEFORE the MsgpackDecoder
    is created, so the decoder uses GREngineCoreRequest's schema.
    GREngineArgs enforces multiprocess_mode=False by default.
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

    Because EngineCoreRequest uses array_like=True, gr_features is
    appended as the last positional element in the serialized array.
    The monkey patch in patch.py replaces EngineCoreRequest globally
    so that MsgpackDecoder uses this subclass schema when decoding.
    """

    # GR feature payload (serializable via msgspec)
    gr_features: GRFeaturesPayload | None = None


__all__ = [
    "GRFeaturesPayload",
    "GREngineCoreRequest",
]
