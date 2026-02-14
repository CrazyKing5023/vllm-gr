# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRWorker: Custom GPU worker for generative recommendation.

Follows the vllm-omni GPUARWorker pattern: extends vLLM's GPUWorker
to use a custom GRModelRunner instead of the default GPUModelRunner.

Injected via worker_cls in GREngineArgs.
"""

import logging

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

logger = init_logger(__name__)


class GRWorker(GPUWorker):
    """
    GPU Worker for generative recommendation.

    Overrides init_device() to replace the default GPUModelRunner
    with GRModelRunner after standard device initialization.
    All other Worker functionality (execute_model, profile, etc.)
    is inherited from GPUWorker.
    """

    def init_device(self):
        """
        Initialize device and replace model_runner with GRModelRunner.

        Delegates all device setup (CUDA init, distributed env, random seed,
        memory snapshot, dtype checks, usage stats) to GPUWorker.init_device(),
        then replaces self.model_runner with GRModelRunner.
        """
        # Let GPUWorker handle all standard device initialization:
        # - NCCL env cleanup
        # - CUDA device setup via current_platform
        # - Distributed environment initialization
        # - Random seed setting
        # - Memory snapshot and validation
        # - Default GPUModelRunner creation
        # - Usage stats reporting
        super().init_device()

        # Replace the default GPUModelRunner with GRModelRunner
        from vllm_gr.worker.gr_model_runner import GRModelRunner
        self.model_runner = GRModelRunner(self.vllm_config, self.device)

        logger.info(
            f"GRWorker: replaced GPUModelRunner with GRModelRunner "
            f"on device {self.device}"
        )
