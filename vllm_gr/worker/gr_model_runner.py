# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner — extends GPUModelRunner for HSTU/DLRM_v3 recommendation.

Overrides:
  _update_states(): decodes GRFeaturesPayload from each new request and
      caches the decoded tensors per request_id.
  _preprocess(): injects GR-specific kwargs (uih_features, candidates_features
      as KeyedJaggedTensors) into model_kwargs so they reach the HSTU
      model's forward() call.
"""

from typing import Any, TYPE_CHECKING

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class GRModelRunner(GPUModelRunner):
    """Model runner that decodes GR features and constructs KJT inputs."""

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
                except Exception as e:
                    logger.error(
                        "Error decoding GR features for %s: %s", req_id, e,
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
        """Convert a GRFeaturesPayload into a dict of tensors for model input.

        Only decodes the fields needed by _build_gr_model_kwargs (KJT
        construction).  Per-request metadata (user_id, num_recommendations,
        original_candidates) is consumed by the output processor via
        GRRequestState, not by the model runner.
        """
        ctx_keys = list(payload.ctx_keys)
        ctx_vals = payload.ctx_values_per_key
        return {
            "uih_keys": list(payload.uih_keys),
            "uih_values_per_key": torch.tensor(
                [val for vals in payload.uih_values_per_key for val in vals],
                dtype=torch.int64
            ),
            "uih_lengths_per_key": torch.tensor(
                payload.uih_lengths_per_key, dtype=torch.int64
            ),
            "cand_keys": list(payload.cand_keys),
            "cand_values_per_key": torch.tensor(
                [val for vals in payload.cand_values_per_key for val in vals],
                dtype=torch.int64
            ),
            "cand_lengths_per_key": torch.tensor(
                payload.cand_lengths_per_key, dtype=torch.int64
            ),
            "ctx_keys": ctx_keys,
            "ctx_values_per_key": torch.tensor(
                [val for vals in ctx_vals for val in vals],
                dtype=torch.int64
            ),
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

    @staticmethod
    def _kjt_batch_func(
        kjt_list: list[KeyedJaggedTensor],
    ) -> KeyedJaggedTensor:
        """
        Batch multiple KeyedJaggedTensors into a single tensor.

        Uses FBGEMM operations for efficient batching and reordering of
        jagged tensor data.

        Args:
            kjt_list: List of KeyedJaggedTensors to batch.

        Returns:
            Batched KeyedJaggedTensor with reordered indices and lengths.
        """
        bs_list = [kjt.stride() for kjt in kjt_list]
        bs = sum(bs_list)
        batched_length = torch.cat([kjt.lengths() for kjt in kjt_list], dim=0)
        batched_indices = torch.cat([kjt.values() for kjt in kjt_list], dim=0)
        
        # Ensure all tensors are on the same device as input KJTs
        device = batched_length.device
        bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(bs_list, device=device)
        ).int()
        batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
        reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
            batched_length, bs_offset, bs
        )
        reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
        reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            batched_offset, batched_indices, reorder_offsets, bs_offset, bs
        )
        out = KeyedJaggedTensor(
            keys=kjt_list[0].keys(),
            lengths=reorder_length.long(),
            values=reorder_indices.long(),
        )
        return out

    def _build_gr_model_kwargs(self) -> dict[str, Any]:
        """Construct KeyedJaggedTensors from decoded features.

        Returns {"uih_features": KJT, "candidates_features": KJT} which are
        passed directly to HSTUForRecommendation.forward() and then to
        DlrmHSTU.forward(uih_features, candidates_features).

        Uses _kjt_batch_func to batch individual request KJTs for efficient
        FBGEMM-based reordering and concatenation.
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
        uih_keys: list[str] = gr_reqs[0][1]["uih_keys"]
        cand_keys: list[str] = gr_reqs[0][1]["cand_keys"]
        ctx_keys: list[str] = gr_reqs[0][1].get("ctx_keys", [])

        # Create individual KJTs for each request
        uih_kjt_list: list[KeyedJaggedTensor] = []
        cand_kjt_list: list[KeyedJaggedTensor] = []

        for _req_id, features in gr_reqs:
            # Build UIH KJT for this request (including ctx features)
            uih_values = features["uih_values_per_key"].to(device)
            ctx_values = features.get(
                "ctx_values_per_key", torch.tensor([], dtype=torch.int64)
            ).to(device)

            uih_lengths = features["uih_lengths_per_key"].to(device)
            ctx_lengths = torch.ones(len(ctx_keys), dtype=torch.int64, device=device)

            req_uih_kjt = KeyedJaggedTensor.from_lengths_sync(
                keys=uih_keys + ctx_keys,
                values=torch.cat([uih_values, ctx_values]),
                lengths=torch.cat([uih_lengths, ctx_lengths]),
            )
            uih_kjt_list.append(req_uih_kjt)

            # Build Cand KJT for this request
            cand_values = features["cand_values_per_key"].to(device)
            cand_lengths = features["cand_lengths_per_key"].to(device)

            req_cand_kjt = KeyedJaggedTensor.from_lengths_sync(
                keys=cand_keys,
                values=cand_values,
                lengths=cand_lengths,
            )
            cand_kjt_list.append(req_cand_kjt)

        # Batch individual KJTs using FBGEMM operations
        uih_kjt = self._kjt_batch_func(uih_kjt_list)
        cand_kjt = self._kjt_batch_func(cand_kjt_list)

        return {
            "uih_features": uih_kjt,
            "candidates_features": cand_kjt,
        }
