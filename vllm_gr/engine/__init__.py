# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Engine-level data structures for generative recommendation.

Defines GRFeaturesPayload and GREngineCoreRequest for transporting
recommendation features through vLLM's EngineCore pipeline.

Serialization note:
    EngineCoreRequest uses array_like=True (positional msgspec encoding).
    GREngineCoreRequest appends gr_features as the last field.  The
    monkey-patch in patch.py replaces EngineCoreRequest in both source
    and consumer modules so that:
      - The client-side MsgpackEncoder serializes gr_features.
      - The subprocess-side MsgpackDecoder(EngineCoreRequest) — which
        now resolves to GREngineCoreRequest — deserializes it correctly.
"""

from typing import Any
import logging

import msgspec

from vllm.v1.engine import EngineCoreRequest

logger = logging.getLogger("vllm_gr.engine")


class GRFeaturesPayload(msgspec.Struct):
    """Msgspec-serializable payload carrying all features for the HSTU model.

    Contains user interaction history (UIH) features and candidate features,
    each represented as parallel lists of keys, values, and lengths.
    """

    uih_keys: list[str]
    uih_values_per_key: list[list[int]]
    uih_lengths_per_key: list[int]

    cand_keys: list[str]
    cand_values_per_key: list[list[int]]
    cand_lengths_per_key: list[int]

    ctx_keys: list[str] = []
    ctx_values_per_key: list[list[int]] = []
    full_uih_keys: list[str] = []

    user_id: str = ""
    original_candidates: list[dict[str, Any]] = []


class GREngineCoreRequest(EngineCoreRequest):
    """EngineCoreRequest extended with a GRFeaturesPayload field.

    Appended as the last positional element in the array_like serialization.
    The monkey patch in patch.py replaces EngineCoreRequest globally so
    MsgpackDecoder uses this subclass schema when decoding.
    """

    gr_features: GRFeaturesPayload | None = None


__all__ = [
    "GRFeaturesPayload",
    "GREngineCoreRequest",
]
