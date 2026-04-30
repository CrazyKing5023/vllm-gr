# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""HSTU model adapter package.

Importing this package triggers the static family registration via
``_family.py`` (layer-1, pure Python, no vLLM dependency).
"""

from vllm_gr.adapters.hstu._family import register_hstu_family

register_hstu_family()
