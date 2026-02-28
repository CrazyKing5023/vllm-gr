# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLM — user-facing entry point for generative recommendation inference.

Inherits from vLLM's LLM class and exposes a recommend() API that accepts
GRPrompt inputs. Internally creates a GRLLMEngine (which has GR-specific
processors) and drives the engine loop to collect GRRequestOutput results.
"""

from typing import Any, Optional, Union, Sequence, List
import time
import logging

from tqdm.auto import tqdm

from vllm.entrypoints.llm import LLM
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.pooling_params import PoolingParams

from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.outputs import GRRequestOutput
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_engine import GRLLMEngine

logger = logging.getLogger("vllm_gr.entrypoint")


class GRLLM(LLM):
    """High-level recommendation interface built on top of vLLM's LLM.

    Does not call super().__init__() because the base LLM creates a
    standard LLMEngine; instead we create a GRLLMEngine directly.
    """

    def __init__(
        self,
        model: str = "",
        embedding_path: Optional[str] = None,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        default_top_k: int = 10,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        device: Optional[str] = None,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        load_model: bool = True,
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        uih_feature_names: Optional[List[str]] = None,
        candidate_feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        use_scheduler = kwargs.pop('use_scheduler', True)
        max_batch_size = kwargs.pop('max_batch_size', 32)
        scheduling_policy = kwargs.pop('scheduling_policy', 'fcfs')

        engine_args = GREngineArgs(
            model=model,
            embedding_path=embedding_path,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            default_top_k=default_top_k,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            embedding_dim=embedding_dim,
            hash_size=hash_size,
            load_model=load_model,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            use_scheduler=use_scheduler,
            gr_max_batch_size=max_batch_size,
            scheduling_policy=scheduling_policy,
            uih_feature_names=(
                uih_feature_names
                or ["item_id", "timestamp", "rating", "action_type"]
            ),
            candidate_feature_names=(
                candidate_feature_names or ["item_id", "category_id"]
            ),
        )

        self.llm_engine = GRLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )
        self.engine_class = type(self.llm_engine)

        self.model = model
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.default_top_k = default_top_k
        self.request_counter = Counter()
        self.default_sampling_params = None
        self.supported_tasks = ("recommendation",)
        self.io_processor = None

        logger.info(
            "GRLLM initialized: model=%s, max_seq_len=%d, "
            "max_candidates=%d, default_top_k=%d",
            model, max_seq_len, max_candidates, default_top_k,
        )

    def recommend(
        self,
        prompts: Union[GRPrompt, Sequence[GRPrompt]],
        *,
        use_tqdm: bool = True,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """Generate recommendation results for one or more GRPrompts."""
        start_time = time.time()

        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        for prompt in prompts:
            if not is_gr_prompt(prompt):
                raise ValueError(
                    f"Expected GRPrompt, got {type(prompt)}. "
                    "GRPrompt must contain 'user_id', 'user_history', "
                    "and 'candidates'."
                )

        request_ids = [f"gr-{next(self.request_counter)}" for _ in prompts]

        for prompt, request_id in zip(prompts, request_ids):
            num_recs = top_k or prompt.get("num_recommendations")
            if num_recs is not None:
                prompt = dict(prompt, num_recommendations=num_recs)

            pooling_params = PoolingParams(task="embed")
            self.llm_engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=pooling_params,
            )

        outputs = self._run_recommendation_engine(
            num_prompts=len(prompts),
            use_tqdm=use_tqdm,
        )

        elapsed = time.time() - start_time
        logger.debug(
            "Recommendation completed in %.3fs for %d prompts",
            elapsed, len(prompts),
        )
        return outputs

    def _run_recommendation_engine(
        self,
        num_prompts: int,
        use_tqdm: bool,
    ) -> List[GRRequestOutput]:
        outputs: List[GRRequestOutput] = []

        if use_tqdm:
            pbar = tqdm(total=num_prompts, desc="Generating recommendations")
        else:
            pbar = None

        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()

            for output in step_outputs:
                if isinstance(output, GRRequestOutput) and output.finished:
                    outputs.append(output)
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

        return outputs

    def recommend_batch(
        self,
        prompts: Sequence[GRPrompt],
        batch_size: int = 32,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """Process prompts in fixed-size batches for large-scale inference."""
        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = self.recommend(batch, top_k=top_k, use_tqdm=False)
            all_outputs.extend(outputs)
        return all_outputs

    def get_engine(self) -> GRLLMEngine:
        return self.llm_engine

    def shutdown(self) -> None:
        if hasattr(self, 'llm_engine') and self.llm_engine is not None:
            self.llm_engine.shutdown()
        logger.info("GRLLM shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
