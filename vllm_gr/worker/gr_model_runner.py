# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner: Custom model runner for generative recommendation.

Extends GPUModelRunner to:
1. Override _update_states() to decode GRFeaturesPayload from
   GRNewRequestData and cache decoded features per request.
2. Override _preprocess() to inject GR-specific model kwargs
   (uih_embeddings, uih_lengths, candidate_embeddings, num_candidates)
   into model_kwargs BEFORE the model forward call.

The base GPUModelRunner.execute_model() calls self.model(**model_kwargs)
directly (no _model_forward hook), so we must inject kwargs via
_preprocess() which builds and returns model_kwargs.

This is the core component that bridges vLLM's execution pipeline
with the HSTU recommendation model.
"""

from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# TorchRec import - optional
try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    _TORCHREC_AVAILABLE = True
except ImportError:
    _TORCHREC_AVAILABLE = False
    KeyedJaggedTensor = None


class GRModelRunner(GPUModelRunner):
    """
    Model runner for generative recommendation (HSTU/DLRM_v3).

    Extends GPUModelRunner to:
    1. Decode GRFeaturesPayload in _update_states and store decoded
       features in per-request cached state.
    2. Inject GR-specific model kwargs via _preprocess override so they
       reach the HSTU model's forward() through model_kwargs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-request GR features stored after decoding in _update_states
        # key: request_id, value: dict of decoded feature tensors/lists
        self._gr_features_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Override _update_states to decode gr_features from GRNewRequestData
    # ------------------------------------------------------------------

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update cached states, extending parent to decode GR features.

        For each new request, if the NewRequestData has gr_features,
        decode the payload and store the decoded feature data
        in self._gr_features_cache[req_id].
        """
        # First, decode GR features from new requests BEFORE calling
        # super()._update_states(), which may remove finished requests.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            gr_features = getattr(new_req_data, "gr_features", None)
            if gr_features is not None:
                try:
                    decoded = self._decode_gr_features(gr_features)
                    self._gr_features_cache[req_id] = decoded
                    logger.debug(
                        f"Decoded GR features for request {req_id}: "
                        f"uih_keys={decoded.get('uih_keys')}, "
                        f"cand_keys={decoded.get('cand_keys')}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error decoding GR features for {req_id}: {e}"
                    )

        # Clean up finished requests from GR features cache
        for req_id in scheduler_output.finished_req_ids:
            self._gr_features_cache.pop(req_id, None)

        # Call parent's _update_states for standard vLLM processing
        super()._update_states(scheduler_output)

    def _decode_gr_features(
        self, payload: Any  # GRFeaturesPayload
    ) -> dict[str, Any]:
        """
        Decode GRFeaturesPayload into a dict of tensors and metadata.

        Converts the serialized int64 lists back into torch tensors.

        Returns:
            dict with keys:
                - uih_keys: list[str]
                - uih_values_per_key: list[torch.Tensor]
                - uih_lengths_per_key: list[torch.Tensor]
                - cand_keys: list[str]
                - cand_values_per_key: list[torch.Tensor]
                - cand_lengths_per_key: list[torch.Tensor]
                - user_id: str
                - num_recommendations: int
                - original_candidates: list[dict]
        """
        decoded = {
            "uih_keys": list(payload.uih_keys),
            "uih_values_per_key": [
                torch.tensor(vals, dtype=torch.int64)
                for vals in payload.uih_values_per_key
            ],
            "uih_lengths_per_key": [
                torch.tensor(lens, dtype=torch.int64)
                for lens in payload.uih_lengths_per_key
            ],
            "cand_keys": list(payload.cand_keys),
            "cand_values_per_key": [
                torch.tensor(vals, dtype=torch.int64)
                for vals in payload.cand_values_per_key
            ],
            "cand_lengths_per_key": [
                torch.tensor(lens, dtype=torch.int64)
                for lens in payload.cand_lengths_per_key
            ],
            "user_id": payload.user_id,
            "num_recommendations": payload.num_recommendations,
            "original_candidates": list(payload.original_candidates),
        }
        return decoded

    # ------------------------------------------------------------------
    # Override _preprocess to inject GR features into model_kwargs
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple:
        """
        Preprocess inputs, then inject GR-specific kwargs into model_kwargs.

        The base GPUModelRunner._preprocess returns a 6-tuple:
            (input_ids, inputs_embeds, positions,
             intermediate_tensors, model_kwargs, ec_connector_output)

        We call super(), then merge GR-specific kwargs into model_kwargs
        so they are passed to self.model(**model_kwargs) in execute_model().
        """
        result = super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )

        # Unpack the 6-tuple; model_kwargs is at index 4
        (input_ids, inputs_embeds, positions,
         interm_tensors, model_kwargs, ec_connector_output) = result

        gr_kwargs = self._build_gr_model_kwargs()
        if gr_kwargs:
            model_kwargs.update(gr_kwargs)
            logger.debug(
                f"Injected GR kwargs into model_kwargs: "
                f"keys={list(gr_kwargs.keys())}"
            )

        return (input_ids, inputs_embeds, positions,
                interm_tensors, model_kwargs, ec_connector_output)

    # ------------------------------------------------------------------
    # Build GR-specific model kwargs
    # ------------------------------------------------------------------

    def _build_gr_model_kwargs(self) -> dict[str, Any]:
        """
        Build GR-specific kwargs for the model's forward() call.

        Iterates over the current batch (self.input_batch.req_ids),
        collects GR features for each request, performs embedding
        lookups, and constructs the batched tensors expected by
        HSTUForRecommendation.forward():
            - uih_embeddings: (total_uih_len, embed_dim) jagged
            - uih_lengths: (batch_size,)
            - candidate_embeddings: (total_cand_len, embed_dim) jagged
            - num_candidates: (batch_size,)

        Returns:
            dict of kwargs, or empty dict if no GR requests in batch.
        """
        if not self._gr_features_cache:
            return {}

        # Check if any request in the current batch has GR features
        batch_req_ids = list(self.input_batch.req_ids)
        gr_reqs = [
            (req_id, self._gr_features_cache[req_id])
            for req_id in batch_req_ids
            if req_id in self._gr_features_cache
        ]

        if not gr_reqs:
            return {}

        device = self.device

        # Get the model (unwrap CUDAGraph wrapper if present)
        model = self.get_model()

        # Collect features and perform embedding lookups
        all_uih_embeddings = []
        all_uih_lengths = []
        all_cand_embeddings = []
        all_num_candidates = []

        for req_id, features in gr_reqs:
            # UIH: use the first key (typically "item_id")
            if features["uih_values_per_key"]:
                uih_item_ids = features["uih_values_per_key"][0].to(device)
                uih_len = len(uih_item_ids)
            else:
                uih_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                uih_len = 0

            # Candidates: use the first key (typically "item_id")
            if features["cand_values_per_key"]:
                cand_item_ids = features["cand_values_per_key"][0].to(device)
                n_cand = len(cand_item_ids)
            else:
                cand_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                n_cand = 0

            # Lookup embeddings from the model's embedding table
            # HSTUForRecommendation uses lookup_embeddings()
            if hasattr(model, "lookup_embeddings"):
                uih_embeds = model.lookup_embeddings(uih_item_ids)
                cand_embeds = model.lookup_embeddings(cand_item_ids)
            elif hasattr(model, "get_item_embeddings"):
                uih_embeds = model.get_item_embeddings(uih_item_ids)
                cand_embeds = model.get_item_embeddings(cand_item_ids)
            else:
                # Fallback: zero embeddings (model should handle this)
                embed_dim = getattr(
                    getattr(model, "config", None),
                    "embedding_dim",
                    128,
                )
                uih_embeds = torch.zeros(
                    uih_len, embed_dim, device=device
                )
                cand_embeds = torch.zeros(
                    n_cand, embed_dim, device=device
                )

            all_uih_embeddings.append(uih_embeds)
            all_uih_lengths.append(uih_len)
            all_cand_embeddings.append(cand_embeds)
            all_num_candidates.append(n_cand)

        # Build jagged tensors (concatenated along dim=0)
        if all_uih_embeddings:
            uih_embeddings = torch.cat(all_uih_embeddings, dim=0)
        else:
            uih_embeddings = torch.zeros(0, device=device)

        if all_cand_embeddings:
            candidate_embeddings = torch.cat(all_cand_embeddings, dim=0)
        else:
            candidate_embeddings = torch.zeros(0, device=device)

        uih_lengths = torch.tensor(
            all_uih_lengths, dtype=torch.int64, device=device
        )
        num_candidates = torch.tensor(
            all_num_candidates, dtype=torch.int64, device=device
        )

        return {
            "uih_embeddings": uih_embeddings,
            "uih_lengths": uih_lengths,
            "candidate_embeddings": candidate_embeddings,
            "num_candidates": num_candidates,
        }
