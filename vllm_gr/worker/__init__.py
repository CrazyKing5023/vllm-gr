# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Worker components for generative recommendation.

Includes:
- GRWorker: Worker for GR model execution
- GRModelRunner: Model runner for HSTU inference
"""

from vllm_gr.worker.gr_model_runner import GRModelRunner
from vllm_gr.worker.gr_worker import GRWorker

__all__ = [
    "GRWorker",
    "GRModelRunner",
]
