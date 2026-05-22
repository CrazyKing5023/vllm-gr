# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""FuXi-alpha model adapter package."""

from vllm_gr.adapters.registry import register_family

register_family(
    model_type="fuxi",
    gpu_worker="vllm_gr.worker.gr_worker.GRWorker",
    npu_worker="vllm_gr.worker.gr_npu_worker.GRNPUWorker",
    defaults_fn="vllm_gr.adapters.fuxi._defaults:apply_fuxi_defaults",
    payload_type="vllm_gr.adapters.fuxi.payload:FuXiFeaturesPayload",
)
