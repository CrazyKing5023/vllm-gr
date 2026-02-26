# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRRequest: Extended vLLM Request with GR feature data.

Follows the vllm-omni OmniRequest pattern: extends vLLM's Request class
to carry GR-specific data (gr_features) through the EngineCore pipeline.
"""

from typing import Any, Optional
from collections.abc import Callable

from vllm.v1.request import Request

from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest


class GRRequest(Request):
    """
    Extended Request that carries GR features through the pipeline.

    When monkey-patched via patch.py, vLLM's EngineCore uses this class
    instead of the base Request, so gr_features flows naturally through
    Scheduler → NewRequestData → ModelRunner.
    """

    def __init__(
        self,
        gr_features: Optional[GRFeaturesPayload] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gr_features: Optional[GRFeaturesPayload] = gr_features

    @classmethod
    def from_engine_core_request(
        cls,
        request: GREngineCoreRequest,
        block_hasher: Optional[Callable] = None,
    ) -> "GRRequest":
        """
        Create GRRequest from GREngineCoreRequest.

        Passes gr_features from the engine core request to the internal
        request object, ensuring data flows through the scheduler.

        Must match the base Request.from_engine_core_request signature.
        In vllm v0.14.1, structured_output_request is computed internally
        by Request.__init__ from sampling_params, so it is NOT passed here.
        """
        return cls(
            # GR-specific field
            gr_features=getattr(request, "gr_features", None),
            # Standard vLLM Request fields (v0.14.1 signature)
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            client_index=getattr(request, "client_index", 0),
            arrival_time=request.arrival_time,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            mm_features=request.mm_features,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )


def is_gr_request(request: Any) -> bool:
    """Check if the request is a GRRequest with GR features."""
    return isinstance(request, GRRequest) and request.gr_features is not None
