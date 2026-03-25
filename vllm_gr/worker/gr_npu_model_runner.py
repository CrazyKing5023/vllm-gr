# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Ascend NPU GR model runner."""

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin


class GRNPUModelRunner(GRFeatureRunnerMixin, NPUModelRunner):
    """NPU model runner that injects GR features into HSTU inputs."""
