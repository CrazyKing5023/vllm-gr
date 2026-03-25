# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Worker components for GR model execution."""

from vllm_gr.worker.gr_worker import GRWorker
from vllm_gr.worker.gr_model_runner import GRModelRunner

__all__ = [
    "GRWorker",
    "GRModelRunner",
]

try:
    from vllm_gr.worker.gr_npu_model_runner import GRNPUModelRunner
    from vllm_gr.worker.gr_npu_worker import GRNPUWorker
    from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin

    __all__.extend([
        "GRNPUModelRunner",
        "GRNPUWorker",
        "GRFeatureRunnerMixin",
    ])
except ImportError:
    pass
