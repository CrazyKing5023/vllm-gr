# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner — extends GPUModelRunner for HSTU/DLRM_v3 recommendation.

Overrides:
  _update_states(): decodes GRFeaturesPayload from each new request and
      caches the decoded tensors per request_id.  Reads ``gr_features``
      from GRNewRequestData (populated via monkey-patch).
  _preprocess(): injects GR-specific kwargs (uih_embeddings, uih_lengths,
      candidate_embeddings, num_candidates) into model_kwargs so they
      reach the HSTU model's forward() call.

The monkey-patch is applied reliably via the vLLM plugin system
(vllm.general_plugins entry point) which runs in every process before
any MsgpackDecoder is constructed.
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
                        "Decoded GR features for request %s: "
                        "uih_keys=%s, cand_keys=%s",
                        req_id,
                        decoded.get("uih_keys"),
                        decoded.get("cand_keys"),
                    )
                except Exception:
                    logger.error(
                        "Error decoding GR features for %s", req_id,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Request %s has no gr_features — monkey-patch may not "
                    "be active.  Run ``from vllm_gr.patch import "
                    "verify_patches; verify_patches()`` to diagnose.",
                    req_id,
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
            "context": getattr(payload, "context", {}) or {},
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

        Returns uih_embeddings, uih_lengths, candidate_embeddings, num_candidates,
        uih_timestamps (when available), and payload_features (when available).
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
        all_uih_timestamps = []
        all_payload_weights = []
        all_context_embeds = []

        context_feature_names = getattr(
            getattr(model, "config", None),
            "context_feature_names",
            ["query_id", "scenario_id"],
        )
        if not isinstance(context_feature_names, list):
            context_feature_names = (
                list(context_feature_names) if context_feature_names else []
            )

        for req_id, features in gr_reqs:
            uih_keys = features.get("uih_keys", [])
            uih_values = features.get("uih_values_per_key", [])
            cand_values = features.get("cand_values_per_key", [])

            # Item IDs: use first key (item_id)
            if uih_values:
                uih_item_ids = uih_values[0].to(device)
                uih_len = len(uih_item_ids)
            else:
                uih_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                uih_len = 0

            if cand_values:
                cand_item_ids = cand_values[0].to(device)
                n_cand = len(cand_item_ids)
            else:
                cand_item_ids = torch.zeros(0, dtype=torch.int64, device=device)
                n_cand = 0

            # Timestamps: extract when "timestamp" in uih_keys (P0 fix)
            if uih_len > 0 and "timestamp" in uih_keys:
                ts_idx = uih_keys.index("timestamp")
                ts_vals = uih_values[ts_idx].to(device=device, dtype=torch.float32)
                uih_ts = ts_vals
            else:
                uih_ts = torch.zeros(uih_len, device=device, dtype=torch.float32)
            cand_ts = torch.zeros(n_cand, device=device, dtype=torch.float32)
            all_uih_timestamps.append(torch.cat([uih_ts, cand_ts], dim=0))

            # Payload: action_weight from rating or action_type (P1 fix)
            # Use rating as primary; fallback to action_type if no rating
            if uih_len > 0 and uih_values:
                pw = None
                if "rating" in uih_keys:
                    r_idx = uih_keys.index("rating")
                    pw = uih_values[r_idx].to(device=device, dtype=torch.float32).unsqueeze(-1)
                elif "action_type" in uih_keys:
                    a_idx = uih_keys.index("action_type")
                    pw = uih_values[a_idx].to(device=device, dtype=torch.float32).unsqueeze(-1)
                if pw is not None:
                    pc = torch.zeros(n_cand, pw.size(-1), device=device, dtype=torch.float32)
                    all_payload_weights.append(torch.cat([pw, pc], dim=0))
                else:
                    all_payload_weights.append(None)
            else:
                all_payload_weights.append(None)

            # Context: embed scalar IDs and broadcast to (seq_len, D) for ContextualPreprocessor
            total_len = uih_len + n_cand
            context = features.get("context") or {}
            if (
                context_feature_names
                and total_len > 0
                and hasattr(model, "lookup_embeddings")
            ):
                ctx_embeds = {}
                for name in context_feature_names:
                    val = context.get(name)
                    if val is None:
                        val = 0
                    try:
                        id_val = int(val)
                    except (TypeError, ValueError):
                        id_val = hash(str(val)) % (1 << 31)
                    ids = torch.tensor(
                        [id_val], dtype=torch.int64, device=device
                    )
                    emb = model.lookup_embeddings(ids)
                    ctx_embeds[name] = emb.expand(total_len, -1).contiguous()
                all_context_embeds.append(ctx_embeds)
            else:
                all_context_embeds.append({})

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
                logger.warning(
                    f"Failed to lookup embeddings for request : {req_id}."
                    f"Using zero embeddings as fallback."
                )
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

        result = {
            "uih_embeddings": uih_embeddings,
            "uih_lengths": uih_lengths,
            "candidate_embeddings": candidate_embeddings,
            "num_candidates": num_candidates,
        }

        # uih_timestamps: (total_uih + total_cand,) float32, aligns with total_embeddings
        if all_uih_timestamps:
            result["uih_timestamps"] = torch.cat(all_uih_timestamps, dim=0)

        # payload_features: action_weight + context embeddings for ContextualPreprocessor
        payload_features: dict[str, torch.Tensor] = {}
        payload_list = []
        for i, pw in enumerate(all_payload_weights):
            if pw is not None:
                payload_list.append(pw)
            else:
                total_len = all_uih_lengths[i] + all_num_candidates[i]
                payload_list.append(
                    torch.zeros(total_len, 1, device=device, dtype=torch.float32)
                )
        if payload_list:
            payload_features["action_weight"] = torch.cat(payload_list, dim=0)

        for name in context_feature_names:
            parts = [req[name] for req in all_context_embeds if name in req]
            if parts:
                payload_features[name] = torch.cat(parts, dim=0)

        if payload_features:
            result["payload_features"] = payload_features

        return result
