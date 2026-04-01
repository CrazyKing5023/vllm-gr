# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Lightweight multitask type definitions shared by config and model code."""

from enum import IntEnum


class MultitaskTaskType(IntEnum):
    BINARY_CLASSIFICATION = 0
    REGRESSION = 1
