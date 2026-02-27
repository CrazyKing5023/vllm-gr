# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GROutputProcessor — extends vLLM's OutputProcessor for recommendation output.

Introduces GRRequestState (extending RequestState) to carry per-request GR
metadata and override make_request_output() to convert raw pooling scores
into ranked GRRequestOutput with Top-K selection.
"""

from typing import Any
import logging

import numpy as np
import torch

from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor as VLLMOutputProcessor,
    RequestOutputCollector,
    RequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest

from vllm_gr.outputs import GRRequestOutput, RecommendedItem

logger = logging.getLogger("vllm_gr.output_processor")


class GRRequestState(RequestState):
    """RequestState extended with GR metadata (user_id, candidates, top-k)."""

    def __init__(self, *args, **kwargs):
        gr_user_id = kwargs.pop("gr_user_id", None)
        gr_num_recommendations = kwargs.pop("gr_num_recommendations", 10)
        gr_original_candidates = kwargs.pop("gr_original_candidates", None)

        super().__init__(*args, **kwargs)

        self.gr_user_id: Any = gr_user_id
        self.gr_num_recommendations: int = gr_num_recommendations
        self.gr_original_candidates: list[dict] = gr_original_candidates or []

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "GRRequestState":
        from vllm.v1.engine.detokenizer import IncrementalDetokenizer
        from vllm.v1.engine.logprobs import LogprobsProcessor

        gr_features = getattr(request, "gr_features", None)
        gr_user_id = gr_features.user_id if gr_features else None
        gr_num_recs = gr_features.num_recommendations if gr_features else 10
        gr_candidates = list(gr_features.original_candidates) if gr_features else []

        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer, request=request)
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer, request=request)
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            gr_user_id=gr_user_id,
            gr_num_recommendations=gr_num_recs,
            gr_original_candidates=gr_candidates,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
    ) -> RequestOutput | PoolingRequestOutput | GRRequestOutput | None:
        """For GR requests, convert pooling scores to ranked GRRequestOutput."""
        if pooling_output is not None and self.gr_original_candidates:
            return self._make_gr_output(pooling_output, finish_reason)

        return super().make_request_output(
            new_token_ids, pooling_output, finish_reason,
            stop_reason, kv_transfer_params, routed_experts)

    def _make_gr_output(
        self,
        pooling_output: torch.Tensor,
        finish_reason: FinishReason | None,
    ) -> GRRequestOutput:
        """Select Top-K candidates by score and build GRRequestOutput."""
        candidates = self.gr_original_candidates
        num_recs = self.gr_num_recommendations
        scores = pooling_output.float()

        top_k = min(num_recs, len(candidates), len(scores))
        if top_k > 0:
            top_scores, top_indices = torch.topk(scores, k=top_k)
            top_scores_list = top_scores.tolist()
            top_indices_list = top_indices.tolist()
        else:
            top_scores_list = []
            top_indices_list = []

        recommendations = []
        for rank, (idx, score) in enumerate(
            zip(top_indices_list, top_scores_list)
        ):
            if idx < len(candidates):
                candidate = candidates[idx]
                recommendations.append(RecommendedItem(
                    item_id=candidate.get("item_id", idx),
                    score=float(score),
                    rank=rank + 1,
                    features=candidate,
                ))

        return GRRequestOutput(
            request_id=self.external_req_id,
            user_id=self.gr_user_id,
            recommendations=recommendations,
            all_scores=pooling_output,
            finished=finish_reason is not None,
        )


class GROutputProcessor(VLLMOutputProcessor):
    """OutputProcessor that uses GRRequestState for recommendation requests.

    Overrides add_request() to instantiate GRRequestState (instead of the
    base RequestState). The parent's process_outputs() loop then naturally
    calls GRRequestState.make_request_output(), which produces
    GRRequestOutput for GR requests without any additional overrides.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        log_stats: bool,
        stream_interval: int = 1,
        default_top_k: int = 10,
    ):
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
        )
        self.default_top_k = default_top_k
        self.request_states: dict[str, GRRequestState] = {}

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        req_state = GRRequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        if self._requests_drained.is_set():
            self._requests_drained.clear()
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        self.external_req_ids[req_state.external_req_id].append(request_id)
