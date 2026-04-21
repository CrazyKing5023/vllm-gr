# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Core data types for generative recommendation.

GRPrompt is the user-facing input format for the GRLLM API.  Feature
keys are dataset-specific and match the names used by the underlying
HSTU model configuration (e.g. ``movie_id`` for MovieLens UIH,
``item_movie_id`` for MovieLens candidates).
"""

from typing import Any, Union, List, Dict

# Pydantic v2 requires typing_extensions.TypedDict on Python < 3.12.
# Keep using typing_extensions to stay compatible with the supported
# Python range (>=3.10,<3.14).
from typing_extensions import NotRequired, TypedDict


class GRPrompt(TypedDict):
    """Input prompt for generative recommendation.

    Feature keys in ``user_history`` and ``candidates`` are
    dataset-specific and must match the model's configured feature
    names.  For UIH the keys correspond to the left side of
    ``merge_uih_candidate_feature_mapping``; for candidates the keys
    correspond to the right side (the model-internal names).

    Example (MovieLens-1M)::

        prompt: GRPrompt = {
            "user_id": 123,
            "user_history": [
                {"movie_id": 1193, "movie_rating": 5,
                 "action_timestamp": 978300760, "item_weights": 1,
                 "dummy_watch_time": 0},
            ],
            "candidates": [
                {"item_movie_id": 1},
                {"item_movie_id": 48},
            ],
            "context": {"user_id": 1, "sex": 0, "age_group": 3},
        }
    """

    user_id: Union[str, int]

    user_history: List[Dict[str, int]]
    """UIH feature dicts keyed by model UIH feature names."""

    candidates: List[Dict[str, int]]
    """Candidate feature dicts keyed by model candidate feature names."""

    context: NotRequired[Dict[str, int]]
    """Contextual features (user-level attributes, one value each)."""


def is_gr_prompt(prompt: Any) -> bool:
    """Return True if *prompt* looks like a valid GRPrompt dict."""
    return (
        isinstance(prompt, dict)
        and "user_id" in prompt
        and "user_history" in prompt
        and "candidates" in prompt
    )
