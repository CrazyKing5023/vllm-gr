# Copyright (c) vllm-GR Authors.
# Original code from Meta Platforms, Inc. (generative-recommenders)
# Licensed under the Apache License, Version 2.0

"""Core model modules for DLRM_v3"""

from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from vllm_gr.models.dlrm_v3.modules.stu import STU, STULayer, STUStack, STULayerConfig
from vllm_gr.models.dlrm_v3.modules.hstu_transducer import HSTUTransducer
from vllm_gr.models.dlrm_v3.modules.positional_encoder import HSTUPositionalEncoder
from vllm_gr.models.dlrm_v3.modules.action_encoder import ActionEncoder
from vllm_gr.models.dlrm_v3.modules.preprocessors import (
    InputPreprocessor,
    ContextualPreprocessor,
)
from vllm_gr.models.dlrm_v3.modules.postprocessors import (
    OutputPostprocessor,
    L2NormPostprocessor,
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from vllm_gr.models.dlrm_v3.modules.multitask_module import (
    MultitaskModule,
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)

__all__ = [
    "ActionEncoder",
    "DlrmHSTU",
    "DlrmHSTUConfig",
    "STU",
    "STULayer",
    "STUStack",
    "STULayerConfig",
    "HSTUTransducer",
    "HSTUPositionalEncoder",
    "InputPreprocessor",
    "ContextualPreprocessor",
    "OutputPostprocessor",
    "L2NormPostprocessor",
    "LayerNormPostprocessor",
    "TimestampLayerNormPostprocessor",
    "MultitaskModule",
    "DefaultMultitaskModule",
    "MultitaskTaskType",
    "TaskConfig",
]

