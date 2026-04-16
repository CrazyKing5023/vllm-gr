# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""GPU GR model runner."""

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin


class GRModelRunner(GRFeatureRunnerMixin, GPUModelRunner):
    """GPU model runner that injects GR features into HSTU inputs."""
    pass
