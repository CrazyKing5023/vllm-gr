# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner — extends GPUModelRunner for HSTU/DLRM_v3 recommendation.

Overrides:
  _update_states(): decodes GRFeaturesPayload from each new request and
      caches the decoded tensors per request_id.
  _preprocess(): injects GR-specific kwargs (uih_embeddings, uih_lengths,
      candidate_embeddings, num_candidates) into model_kwargs so they
      reach the HSTU model's forward() call.
"""

from typing import Any, TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    _TORCHREC_AVAILABLE = True
except ImportError:
    _TORCHREC_AVAILABLE = False
    KeyedJaggedTensor = None


class GRModelRunner(GPUModelRunner):
    """Model runner that decodes GR features and injects them into model_kwargs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gr_features_cache: dict[str, dict[str, Any]] = {}

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
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

        for req_id in scheduler_output.finished_req_ids:
            self._gr_features_cache.pop(req_id, None)

        super()._update_states(scheduler_output)

    def _decode_gr_features(self, payload: Any) -> dict[str, Any]:
        """Convert a GRFeaturesPayload into a dict of tensors and metadata."""
        return {
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

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple:
        """Call parent _preprocess, then merge GR kwargs into model_kwargs."""
        result = super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )

        (input_ids, inputs_embeds, positions,
         interm_tensors, model_kwargs, ec_connector_output) = result

        gr_kwargs = self._build_gr_model_kwargs()
        if gr_kwargs:
            model_kwargs.update(gr_kwargs)

        return (input_ids, inputs_embeds, positions,
                interm_tensors, model_kwargs, ec_connector_output)

    def _build_gr_model_kwargs(self) -> dict[str, Any]:
        """Build batched GR tensors expected by HSTUForRecommendation.forward().

        Returns uih_embeddings, uih_lengths, candidate_embeddings, and
        num_candidates, or an empty dict if no GR requests are in the batch.
        """
        if not self._gr_features_cache:
            return {}

        batch_req_ids = list(self.input_batch.req_ids)
        gr_reqs = [
            (req_id, self._gr_features_cache[req_id])
            for req_id in batch_req_ids
            if req_id in self._gr_features_cache
        ]

        if not gr_reqs:
            return {}

        device = self.device
        model = self.get_model()

        all_uih_embeddings = []
        all_uih_lengths = []
        all_cand_embeddings = []
        all_num_candidates = []

        for req_id, features in gr_reqs:
            if features["uih_values_per_key"]:
                uih_item_ids = features["uih_values_per_key"][0].to(device)
                uih_len = len(uih_item_ids)
            else:
                uih_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                uih_len = 0

            if features["cand_values_per_key"]:
                cand_item_ids = features["cand_values_per_key"][0].to(device)
                n_cand = len(cand_item_ids)
            else:
                cand_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                n_cand = 0

            if hasattr(model, "lookup_embeddings"):
                uih_embeds = model.lookup_embeddings(uih_item_ids)
                cand_embeds = model.lookup_embeddings(cand_item_ids)
            elif hasattr(model, "get_item_embeddings"):
                uih_embeds = model.get_item_embeddings(uih_item_ids)
                cand_embeds = model.get_item_embeddings(cand_item_ids)
            else:
                embed_dim = getattr(
                    getattr(model, "config", None),
                    "embedding_dim",
                    128,
                )
                uih_embeds = torch.zeros(uih_len, embed_dim, device=device)
                cand_embeds = torch.zeros(n_cand, embed_dim, device=device)

            all_uih_embeddings.append(uih_embeds)
            all_uih_lengths.append(uih_len)
            all_cand_embeddings.append(cand_embeds)
            all_num_candidates.append(n_cand)

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
