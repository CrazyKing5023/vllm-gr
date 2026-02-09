# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Engine components for generative recommendation.

Includes:
- GRLLMEngine: Main engine class
- GRProcessor: Input processor (GRPrompt -> GRRequest)
- GROutputProcessor: Output processor (model output -> GRRequestOutput)
- GREngineArgs: Engine arguments
- GRModelRunner: Model runner for real HSTU inference
"""

# Processors
from vllm_gr.engine.gr_processor import GRProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs

# Engine arguments
from vllm_gr.engine.arg_utils import GREngineArgs

# Main engine
from vllm_gr.engine.gr_engine import GRLLMEngine

# Model runner (requires torch, which is a mandatory dependency)
from vllm_gr.engine.model_runner import GRModelRunner

__all__ = [
    # Processors
    "GRProcessor",
    "GROutputProcessor",
    "GRProcessedOutputs",
    
    # Engine
    "GREngineArgs",
    "GRLLMEngine",
    
    # Model runner
    "GRModelRunner",
]
