# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Core components for generative recommendation.

Includes:
- GRScheduler: Scheduler for GR requests (implements vLLM SchedulerInterface)
- GRSchedulerConfig: Scheduler configuration
"""

from vllm_gr.core.gr_scheduler import GRScheduler, GRSchedulerConfig

__all__ = [
    "GRScheduler",
    "GRSchedulerConfig",
]
