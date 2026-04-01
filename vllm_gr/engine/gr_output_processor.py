# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GROutputProcessor — extends vLLM's OutputProcessor for recommendation output.

Introduces GRRequestState (extending RequestState) to carry per-request GR
metadata and override make_request_output() to convert raw pooling scores
into candidate-score GRRequestOutput objects.
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
    """RequestState extended with GR metadata (user_id, candidates, tasks)."""

    def __init__(self, *args, **kwargs):
        is_gr_request = kwargs.pop("is_gr_request", False)
        gr_user_id = kwargs.pop("gr_user_id", None)
        gr_original_candidates = kwargs.pop("gr_original_candidates", None)
        gr_task_names = kwargs.pop("gr_task_names", None)
        gr_primary_task = kwargs.pop("gr_primary_task", "")
        gr_primary_id_key = kwargs.pop("gr_primary_id_key", "item_id")

        super().__init__(*args, **kwargs)

        self.is_gr_request: bool = is_gr_request
        self.gr_user_id: Any = gr_user_id
        self.gr_original_candidates: list[dict] = gr_original_candidates or []
        self.gr_task_names: list[str] = gr_task_names or []
        self.gr_primary_task: str = gr_primary_task
        self.gr_primary_id_key: str = gr_primary_id_key

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
        task_names: list[str] | None = None,
        primary_task: str = "",
        primary_id_key: str = "item_id",
    ) -> "GRRequestState":
        from vllm.v1.engine.detokenizer import IncrementalDetokenizer
        from vllm.v1.engine.logprobs import LogprobsProcessor

        gr_features = getattr(request, "gr_features", None)
        gr_user_id = gr_features.user_id if gr_features else None
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
            is_gr_request=gr_features is not None,
            gr_user_id=gr_user_id,
            gr_original_candidates=gr_candidates,
            gr_primary_id_key=primary_id_key,
            gr_task_names=task_names or [],
            gr_primary_task=primary_task,
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
        """For GR requests, convert pooling scores to GRRequestOutput."""

        if self.is_gr_request:
            return self._make_gr_output(pooling_output, finish_reason)

        return super().make_request_output(
            new_token_ids, pooling_output, finish_reason,
            stop_reason, kv_transfer_params, routed_experts)

    def _make_gr_output(
        self,
        pooling_output: torch.Tensor,
        finish_reason: FinishReason | None,
    ) -> GRRequestOutput:
        """Build a GRRequestOutput in the original candidate order.

        The HSTU pooling path provides task-major score tensors with shape
        ``(num_tasks, num_candidates)``. The primary task score view becomes
        the item score and all per-task rows are attached when task metadata is
        available.
        """

        if not self.gr_original_candidates or pooling_output is None:
            return GRRequestOutput(
                request_id=self.external_req_id,
                user_id=self.gr_user_id,
                recommendations=[],
                all_scores=pooling_output,
                finished=finish_reason is not None,
            )

        candidates = self.gr_original_candidates
        scores = pooling_output.float()

        if scores.dim() != 2:
            raise ValueError(
                "GR pooling output must be 2D "
                "(num_tasks, num_candidates): "
                f"shape={tuple(scores.shape)}"
            )

        all_task_scores: dict[str, list[float]] | None = None

        if self.gr_task_names:
            if self.gr_primary_task and self.gr_primary_task in self.gr_task_names:
                primary_idx = self.gr_task_names.index(self.gr_primary_task)
            else:
                primary_idx = 0
            primary_scores = scores[primary_idx]
            all_task_scores = {
                name: scores[i].tolist()
                for i, name in enumerate(self.gr_task_names)
            }
        else:
            primary_scores = scores[0]

        pid_key = self.gr_primary_id_key
        recommendations = []
        if len(candidates) != len(primary_scores):
            raise ValueError(
                "GR candidate/score count mismatch: "
                f"candidates={len(candidates)}, scores={len(primary_scores)}"
            )

        for idx in range(len(candidates)):
            candidate = candidates[idx]
            item_task_scores = None
            if all_task_scores is not None:
                item_task_scores = {
                    t: all_task_scores[t][idx] for t in all_task_scores
                }
            recommendations.append(RecommendedItem(
                item_id=candidate.get(pid_key, idx),
                score=float(primary_scores[idx]),
                features=candidate,
                task_scores=item_task_scores,
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
        task_names: list[str] | None = None,
        primary_task: str = "",
        primary_id_key: str = "item_id",
    ):
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
        )
        self.task_names: list[str] = task_names or []
        self.primary_task: str = primary_task
        self.primary_id_key: str = primary_id_key
        self.request_states: dict[str, GRRequestState] = {}

    @property
    def num_pending_requests(self) -> int:
        return len(self.request_states)

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
            task_names=self.task_names,
            primary_task=self.primary_task,
            primary_id_key=self.primary_id_key,
        )
        if self._requests_drained.is_set():
            self._requests_drained.clear()
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        self.external_req_ids[req_state.external_req_id].append(request_id)
