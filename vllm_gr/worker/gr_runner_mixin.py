# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Shared GR-specific model runner logic."""

from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_gr.gr_schema import (
    attach_schema_fingerprint,
    validate_context_group,
    validate_feature_group,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def _concat_1d_int_tensors(
    tensor_parts: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not tensor_parts:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(tensor_parts, dim=0)


class GRFeatureRunnerMixin:
    """Inject GR feature decode and KJT model-kwargs building into a runner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gr_features_cache: dict[str, dict[str, Any]] = {}
        self._gr_feature_errors: dict[str, str] = {}

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            gr_features = getattr(new_req_data, "gr_features", None)
            if gr_features is not None:
                try:
                    decoded = self._decode_gr_features(gr_features)
                    self._gr_features_cache[req_id] = decoded
                    self._gr_feature_errors.pop(req_id, None)
                    logger.debug(
                        "Decoded GR features for request %s: uih_keys=%s, cand_keys=%s",
                        req_id,
                        decoded.get("uih_keys"),
                        decoded.get("cand_keys"),
                    )
                except Exception as e:
                    self._gr_features_cache.pop(req_id, None)
                    self._gr_feature_errors[req_id] = str(e)
                    logger.error(
                        "Error decoding GR features for %s: %s",
                        req_id,
                        e,
                    )
            else:
                logger.warning(
                    "Request %s has no gr_features; monkey-patch may not be active.",
                    req_id,
                )

        for req_id in scheduler_output.finished_req_ids:
            self._gr_features_cache.pop(req_id, None)
            self._gr_feature_errors.pop(req_id, None)

        super()._update_states(scheduler_output)

    def _decode_gr_features(self, payload: Any) -> dict[str, Any]:
        uih_keys = list(payload.uih_keys)
        cand_keys = list(payload.cand_keys)
        ctx_keys = list(payload.ctx_keys)
        (
            uih_keys,
            cand_keys,
            ctx_keys,
            schema_fingerprint,
        ) = attach_schema_fingerprint(
            uih_keys=uih_keys,
            cand_keys=cand_keys,
            ctx_keys=ctx_keys,
        )
        validate_feature_group(
            "uih",
            uih_keys,
            list(payload.uih_values_per_key),
            list(payload.uih_lengths_per_key),
        )
        validate_feature_group(
            "cand",
            cand_keys,
            list(payload.cand_values_per_key),
            list(payload.cand_lengths_per_key),
        )
        validate_context_group(
            ctx_keys,
            list(payload.ctx_values_per_key),
        )

        full_uih_keys = list(getattr(payload, "full_uih_keys", []) or (uih_keys + ctx_keys))
        combined_uih_keys = uih_keys + ctx_keys
        if (
            len(full_uih_keys) != len(combined_uih_keys)
            or set(full_uih_keys) != set(combined_uih_keys)
        ):
            raise ValueError(
                "GR payload full_uih_keys do not match decoded UIH/context keys: "
                f"full_uih_keys={full_uih_keys!r}, combined_keys={combined_uih_keys!r}"
            )

        return {
            "uih_keys": uih_keys,
            "uih_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.uih_values_per_key
            ],
            "uih_lengths_per_key": torch.tensor(
                payload.uih_lengths_per_key,
                dtype=torch.int64,
            ),
            "cand_keys": cand_keys,
            "cand_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.cand_values_per_key
            ],
            "cand_lengths_per_key": torch.tensor(
                payload.cand_lengths_per_key,
                dtype=torch.int64,
            ),
            "ctx_keys": ctx_keys,
            "ctx_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.ctx_values_per_key
            ],
            "full_uih_keys": full_uih_keys,
            "schema_fingerprint": schema_fingerprint,
        }

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple:
        result = super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )

        (
            input_ids,
            inputs_embeds,
            positions,
            interm_tensors,
            model_kwargs,
            ec_connector_output,
        ) = result

        gr_kwargs = self._build_gr_model_kwargs()
        if gr_kwargs:
            model_kwargs.update(gr_kwargs)

        return (
            input_ids,
            inputs_embeds,
            positions,
            interm_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _build_gr_model_kwargs(self) -> dict[str, Any]:
        batch_req_ids = list(self.input_batch.req_ids)
        failed_req_ids = [
            req_id for req_id in batch_req_ids
            if req_id in self._gr_feature_errors
        ]
        if failed_req_ids:
            error_text = ", ".join(
                f"{req_id}: {self._gr_feature_errors[req_id]}"
                for req_id in failed_req_ids
            )
            raise ValueError(
                "GR feature decode failed for scheduled requests: "
                f"{error_text}"
            )

        if not self._gr_features_cache:
            return {}

        gr_reqs = [
            (req_id, self._gr_features_cache[req_id])
            for req_id in batch_req_ids
            if req_id in self._gr_features_cache
        ]
        if not gr_reqs:
            return {}
        if len(gr_reqs) != len(batch_req_ids):
            missing_req_ids = [
                req_id for req_id in batch_req_ids
                if req_id not in self._gr_features_cache
            ]
            raise ValueError(
                "Incomplete GR feature batch: missing decoded features for "
                f"requests {missing_req_ids}"
            )

        self._validate_batch_schema(gr_reqs)

        device = torch.device("cpu")
        uih_keys: list[str] = gr_reqs[0][1]["uih_keys"]
        cand_keys: list[str] = gr_reqs[0][1]["cand_keys"]
        ctx_keys: list[str] = gr_reqs[0][1]["ctx_keys"]
        full_uih_keys: list[str] = gr_reqs[0][1].get(
            "full_uih_keys",
            uih_keys + ctx_keys,
        )
        num_uih_keys = len(uih_keys)
        num_cand_keys = len(cand_keys)
        num_ctx_keys = len(ctx_keys)

        uih_values_by_key: list[list[torch.Tensor]] = [
            [] for _ in range(num_uih_keys)
        ]
        uih_lengths_by_key: list[list[int]] = [[] for _ in range(num_uih_keys)]
        cand_values_by_key: list[list[torch.Tensor]] = [
            [] for _ in range(num_cand_keys)
        ]
        cand_lengths_by_key: list[list[int]] = [[] for _ in range(num_cand_keys)]
        ctx_values_by_key: list[list[torch.Tensor]] = [
            [] for _ in range(num_ctx_keys)
        ]

        for _req_id, features in gr_reqs:
            uih_lengths = features["uih_lengths_per_key"]
            cand_lengths = features["cand_lengths_per_key"]

            for key_idx in range(num_uih_keys):
                values = features["uih_values_per_key"][key_idx].to(device)
                uih_values_by_key[key_idx].append(values)
                uih_lengths_by_key[key_idx].append(
                    int(uih_lengths[key_idx].item())
                )

            for key_idx in range(num_cand_keys):
                values = features["cand_values_per_key"][key_idx].to(device)
                cand_values_by_key[key_idx].append(values)
                cand_lengths_by_key[key_idx].append(
                    int(cand_lengths[key_idx].item())
                )

            for key_idx in range(num_ctx_keys):
                values = features["ctx_values_per_key"][key_idx].reshape(-1).to(device)
                ctx_values_by_key[key_idx].append(values)

        uih_key_to_idx = {key: idx for idx, key in enumerate(uih_keys)}
        ctx_key_to_idx = {key: idx for idx, key in enumerate(ctx_keys)}
        all_uih_value_parts: list[torch.Tensor] = []
        all_uih_lengths: list[int] = []
        batch_size = len(gr_reqs)

        for key in full_uih_keys:
            if key in uih_key_to_idx:
                key_idx = uih_key_to_idx[key]
                values_per_request = uih_values_by_key[key_idx]
                if values_per_request:
                    all_uih_value_parts.append(torch.cat(values_per_request, dim=0))
                all_uih_lengths.extend(uih_lengths_by_key[key_idx])
                continue

            if key in ctx_key_to_idx:
                key_idx = ctx_key_to_idx[key]
                values_per_request = ctx_values_by_key[key_idx]
                if values_per_request:
                    all_uih_value_parts.append(torch.cat(values_per_request, dim=0))
                all_uih_lengths.extend([1] * batch_size)
                continue

            raise ValueError(
                "Full UIH schema contains unknown key not present in GR payload: "
                f"{key!r}"
            )

        uih_values = _concat_1d_int_tensors(
            all_uih_value_parts,
            device=device,
        )
        uih_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=full_uih_keys,
            values=uih_values,
            lengths=torch.tensor(all_uih_lengths, dtype=torch.int64, device=device),
        )

        cand_values = _concat_1d_int_tensors(
            [
                _concat_1d_int_tensors(values_per_key, device=device)
                for values_per_key in cand_values_by_key
            ],
            device=device,
        )
        cand_lengths = torch.tensor(
            [
                length
                for lengths_per_key in cand_lengths_by_key
                for length in lengths_per_key
            ],
            dtype=torch.int64,
            device=device,
        )
        cand_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=cand_keys,
            values=cand_values,
            lengths=cand_lengths,
        )

        return {
            "uih_features": uih_kjt,
            "candidates_features": cand_kjt,
        }

    def _pool(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: Any,
    ) -> Any:
        """Handle GR pooling outputs that are already split per request.

        vLLM's generic pooling path assumes `hidden_states` is a single tensor
        indexed by scheduled token positions. HSTU returns recommendation
        scores, and in the GR path those scores may already be partitioned as
        one tensor per request. In that case we construct `pooler_output`
        directly instead of delegating to the token-centric base implementation.
        """
        if isinstance(hidden_states, torch.Tensor):
            return super()._pool(
                hidden_states,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                kv_connector_output,
            )

        from vllm.v1.outputs import ModelRunnerOutput

        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        per_req_hidden_states = list(hidden_states[:num_reqs])
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pool_device = next(
            (
                tensor.device
                for tensor in per_req_hidden_states
                if isinstance(tensor, torch.Tensor)
            ),
            torch.device("cpu"),
        )
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=pool_device,
        )

        raw_pooler_output = self.model.pooler(
            hidden_states=per_req_hidden_states,
            pooling_metadata=pooling_metadata,
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if isinstance(raw_pooler_output, torch.Tensor):
            raw_outputs: list[torch.Tensor | None] = [raw_pooler_output]
        else:
            raw_outputs = list(raw_pooler_output)

        if len(raw_outputs) != num_reqs:
            raise ValueError(
                "GR pooling output count mismatch: "
                f"num_reqs={num_reqs}, outputs={len(raw_outputs)}"
            )

        raw_outputs = [
            None
            if tensor is None
            else tensor.to("cpu", non_blocking=True)
            for tensor in raw_outputs
        ]
        model_runner_output.pooler_output = [
            output if include else None
            for output, include in zip(raw_outputs, finished_mask)
        ]

        sync_device = getattr(self, "_sync_device", None)
        if callable(sync_device):
            sync_device()

        return model_runner_output

    def _validate_batch_schema(
        self,
        gr_reqs: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Fail fast when mixed GR requests carry different schemas."""
        base_req_id, base_features = gr_reqs[0]
        expected = str(base_features["schema_fingerprint"])
        expected_full_uih_keys = list(
            base_features.get(
                "full_uih_keys",
                base_features["uih_keys"] + base_features["ctx_keys"],
            )
        )
        mismatches: list[str] = []

        for req_id, features in gr_reqs[1:]:
            actual = str(features["schema_fingerprint"])
            actual_full_uih_keys = list(
                features.get(
                    "full_uih_keys",
                    features["uih_keys"] + features["ctx_keys"],
                )
            )
            if actual != expected or actual_full_uih_keys != expected_full_uih_keys:
                mismatch_desc = f"{req_id}={actual}"
                if actual_full_uih_keys != expected_full_uih_keys:
                    mismatch_desc += (
                        " full_uih_keys="
                        f"{actual_full_uih_keys!r}"
                    )
                mismatches.append(mismatch_desc)

        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(
                "Mixed GR batch schema mismatch: "
                f"expected {base_req_id}={expected}; got {mismatch_text}"
            )
