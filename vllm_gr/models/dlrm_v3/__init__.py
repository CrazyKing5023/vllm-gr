# Copyright (c) vllm-GR Authors.
# Original code from Meta Platforms, Inc. (generative-recommenders)
# Licensed under the Apache License, Version 2.0

"""
DLRM_v3 Model - Ported from generative-recommenders

This package contains the HSTU-based generative recommender model,
fully transplanted from Meta's generative-recommenders repository.
"""

from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from vllm_gr.models.dlrm_v3.modules.stu import STU, STULayer, STUStack, STULayerConfig
from vllm_gr.models.dlrm_v3.modules.hstu_transducer import HSTUTransducer

__all__ = [
    "DlrmHSTU",
    "DlrmHSTUConfig",
    "STU",
    "STULayer",
    "STUStack",
    "STULayerConfig",
    "HSTUTransducer",
]

