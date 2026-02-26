# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRNewRequestData: Extended scheduler output for GR features.

Follows the vllm-omni OmniNewRequestData pattern: extends vLLM's
NewRequestData to carry GR feature data from the Scheduler to the
ModelRunner via SchedulerOutput.
"""

from dataclasses import dataclass
from typing import Optional

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request

from vllm_gr.engine import GRFeaturesPayload


@dataclass
class GRNewRequestData(NewRequestData):
    """
    Extended NewRequestData carrying GR features.

    When the Scheduler creates output for new requests, this class
    ensures gr_features flows from GRRequest → GRNewRequestData →
    GRModelRunner._update_states().
    """

    # GR feature payload (serialized, msgspec-compatible)
    gr_features: GRFeaturesPayload | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "GRNewRequestData":
        """
        Create GRNewRequestData from a GRRequest.

        Extracts gr_features from the request and includes it in the
        scheduler output so that ModelRunner can access it.
        v0.14.1 added prefill_token_ids for v2 model runner support.
        """
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prefill_token_ids=prefill_token_ids,
            # GR-specific: extract gr_features from GRRequest
            gr_features=getattr(request, "gr_features", None),
        )
