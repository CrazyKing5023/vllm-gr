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
            "ctx_keys": ctx_keys,
            "ctx_values_per_key": [
                torch.tensor(vals, dtype=torch.int64) for vals in ctx_vals
            ],
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
        """Construct KeyedJaggedTensors from decoded features.

        Returns {"uih_features": KJT, "candidates_features": KJT} which are
        passed directly to HSTUForRecommendation.forward() and then to
        DlrmHSTU.forward(uih_features, candidates_features).

        KJT layout (TorchRec convention):
          batch_size = number of users (requests)
          For K keys and B batch elements:
            values:  all values for key_0 (user_0, user_1, …), then key_1, …
            lengths: K*B entries — [key0_user0, key0_user1, …, key1_user0, …]
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
        num_uih_keys = len(uih_keys)
        num_cand_keys = len(cand_keys)
        num_ctx_keys = len(ctx_keys)

        uih_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_uih_keys)]
        uih_lengths_by_key: list[list[int]] = [[] for _ in range(num_uih_keys)]
        cand_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_cand_keys)]
        cand_lengths_by_key: list[list[int]] = [[] for _ in range(num_cand_keys)]
        ctx_values_by_key: list[list[torch.Tensor]] = [[] for _ in range(num_ctx_keys)]

        for _req_id, features in gr_reqs:
            for k in range(num_uih_keys):
                vals = features["uih_values_per_key"][k].to(device)
                uih_values_by_key[k].append(vals)
                uih_lengths_by_key[k].append(len(vals))

            for k in range(num_cand_keys):
                vals = features["cand_values_per_key"][k].to(device)
                cand_values_by_key[k].append(vals)
                cand_lengths_by_key[k].append(len(vals))

            ctx_vals_list = features.get("ctx_values_per_key", [])
            for k in range(num_ctx_keys):
                vals = ctx_vals_list[k].to(device)
                ctx_values_by_key[k].append(vals)

        # Contextual features (user-level, length=1 per batch element) are
        # appended to the UIH KJT so the EmbeddingCollection can look them up.
        all_uih_keys = uih_keys + ctx_keys

        all_uih_values_parts: list[torch.Tensor] = []
        all_uih_lengths: list[int] = []

        for k in range(num_uih_keys):
            all_uih_values_parts.append(torch.cat(uih_values_by_key[k]))
            all_uih_lengths.extend(uih_lengths_by_key[k])

        batch_size = len(gr_reqs)
        for k in range(num_ctx_keys):
            all_uih_values_parts.append(torch.cat(ctx_values_by_key[k]))
            all_uih_lengths.extend([1] * batch_size)

        uih_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=all_uih_keys,
            values=torch.cat(all_uih_values_parts),
            lengths=torch.tensor(all_uih_lengths, dtype=torch.int64, device=device),
        )
        cand_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=cand_keys,
            values=torch.cat([
                torch.cat(cand_values_by_key[k]) for k in range(num_cand_keys)
            ]),
            lengths=torch.tensor(
                [l for k in range(num_cand_keys) for l in cand_lengths_by_key[k]],
                dtype=torch.int64, device=device,
            ),
        )

        return {
            "uih_features": uih_kjt,
            "candidates_features": cand_kjt,
        }
