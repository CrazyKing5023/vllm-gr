# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRWorker — extends GPUWorker to use GRModelRunner.

Injected via worker_cls in GREngineArgs so vLLM's EngineCore
instantiates this worker instead of the default GPUWorker.
"""

import logging

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

logger = init_logger(__name__)


class GRWorker(GPUWorker):
    """GPU worker that replaces the default model runner with GRModelRunner."""

    def init_device(self):
        super().init_device()

        from vllm_gr.worker.gr_model_runner import GRModelRunner
        self.model_runner = GRModelRunner(self.vllm_config, self.device)

        logger.info(
            f"GRWorker: using GRModelRunner on device {self.device}"
        )
