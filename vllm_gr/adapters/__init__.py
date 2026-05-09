# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""GR model adapter framework.

Provides ``GRModelAdapter`` base class and capability protocols for
model-specific recommendation inference pipelines.
"""

from vllm_gr.adapters.base import GRModelAdapter

__all__ = [
    "GRModelAdapter",
]
