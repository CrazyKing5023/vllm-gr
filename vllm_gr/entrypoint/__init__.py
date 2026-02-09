# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Entry points for vllm-GR"""

from vllm_gr.entrypoint.gr import GRLLM, RecommendationResponse

__all__ = ["GRLLM", "RecommendationResponse"]

