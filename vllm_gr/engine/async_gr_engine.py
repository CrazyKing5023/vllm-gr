# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerFactory

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor
from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.outputs import GRRequestOutput


class GRAsyncLLM(AsyncLLM):
    """GR async engine client for online recommendation serving."""

    gr_config: GRConfig

    def _replace_processors(self) -> None:
        hf_config = self.vllm_config.model_config.hf_config
        cfg = self.gr_config

        uih_feature_names = (
            hf_config.hstu_uih_feature_names or hf_config.user_embedding_feature_names
        )
        candidate_feature_names = (
            hf_config.hstu_candidate_feature_names
            or hf_config.item_embedding_feature_names
        )
        configured_context_feature_names = list(
            dict.fromkeys(
                list(getattr(hf_config, "contextual_feature_to_max_length", {}).keys())
                + list(
                    getattr(hf_config, "contextual_feature_to_min_uih_length", {}).keys()
                )
            )
        )
        context_feature_names = [
            key for key in uih_feature_names if key in set(configured_context_feature_names)
        ]
        primary_id_feature_name = hf_config.uih_post_id_feature_name
        candidate_primary_id_key = self._resolve_candidate_primary_id_key(
            hf_config=hf_config,
            candidate_feature_names=candidate_feature_names,
        )
        task_names = list(hf_config.task_names)
        primary_task = task_names[0] if task_names else ""

        self.input_processor = GRInputProcessor(
            vllm_config=self.vllm_config,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_candidates=cfg.max_candidates,
            uih_feature_names=uih_feature_names,
            candidate_feature_names=candidate_feature_names,
            primary_id_feature_name=primary_id_feature_name,
            full_uih_feature_names=uih_feature_names,
            context_feature_names=context_feature_names,
        )

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = GROutputProcessor(
            tokenizer=self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=stream_interval,
            task_names=task_names,
            primary_task=primary_task,
            primary_id_key=candidate_primary_id_key,
        )

    def _restart_output_handler_with_gr_processors(self) -> None:
        """Restart the background output handler after replacing processors."""
        task = getattr(self, "output_handler", None)
        if task is not None:
            task.cancel()
            self.output_handler = None

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        self._run_output_handler()

    @staticmethod
    def _resolve_candidate_primary_id_key(
        *,
        hf_config,
        candidate_feature_names: list[str],
    ) -> str:
        explicit_candidate_id_key = str(
            getattr(hf_config, "candidate_primary_id_feature_name", "") or ""
        )
        if explicit_candidate_id_key:
            return explicit_candidate_id_key

        merge_mapping = list(
            getattr(hf_config, "merge_uih_candidate_feature_mapping", []) or []
        )
        primary_uih_id_key = getattr(hf_config, "uih_post_id_feature_name", "")
        for uih_key, candidate_key in merge_mapping:
            if uih_key == primary_uih_id_key:
                return candidate_key

        raise ValueError(
            "Unable to resolve candidate primary id feature. Configure "
            "`candidate_primary_id_feature_name` explicitly or provide a "
            "`merge_uih_candidate_feature_mapping` entry that maps "
            f"{primary_uih_id_key!r} to one of {candidate_feature_names!r}."
        )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        gr_config: GRConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "GRAsyncLLM":
        base = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=enable_log_requests,
            aggregate_engine_logging=aggregate_engine_logging,
            disable_log_stats=disable_log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )
        base.__class__ = cls
        gr_async_llm = cast(GRAsyncLLM, base)
        gr_async_llm.gr_config = gr_config
        gr_async_llm._replace_processors()
        gr_async_llm._restart_output_handler_with_gr_processors()
        return gr_async_llm

    async def encode(
        self,
        prompt: GRPrompt,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict | None = None,
    ) -> AsyncGenerator[GRRequestOutput, None]:
        if not is_gr_prompt(prompt):
            raise TypeError(
                "GRAsyncLLM.encode expects GRPrompt input; "
                f"got {type(prompt).__name__}"
            )
        if pooling_params.task != "embed":
            raise ValueError(
                "GRAsyncLLM.encode requires PoolingParams(task='embed'); "
                f"got task={pooling_params.task!r}"
            )

        async for out in super().encode(
            prompt=prompt,
            pooling_params=pooling_params,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            truncate_prompt_tokens=truncate_prompt_tokens,
            tokenization_kwargs=tokenization_kwargs,
        ):
            if not isinstance(out, GRRequestOutput):
                raise TypeError(
                    "GRAsyncLLM expected GRRequestOutput from GROutputProcessor; "
                    f"got {type(out).__name__}"
                )
            yield out


class GRAsyncLLMEngine(GRAsyncLLM):
    """Alias class to align with AsyncLLMEngine naming."""

