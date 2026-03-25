# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Ascend NPU worker for GR/HSTU execution."""

from vllm.logger import init_logger
from vllm_ascend.worker.worker import NPUWorker

logger = init_logger(__name__)


class GRNPUWorker(NPUWorker):
    """NPU worker that replaces the default runner with GRNPUModelRunner."""

    def init_device(self):
        super().init_device()

        from vllm_gr.patch import re_apply_patches
        re_apply_patches()

        from vllm_gr.worker.gr_npu_model_runner import GRNPUModelRunner

        self.model_runner = GRNPUModelRunner(self.vllm_config, self.device)
        logger.info(
            "GRNPUWorker: using GRNPUModelRunner on device %s",
            self.device,
        )
