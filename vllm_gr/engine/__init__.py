# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Engine components for generative recommendation.

Core Components:
- GRLLMEngine: Main engine orchestrating inference
- GRScheduler: Request scheduling and batching
- GRWorker: Model execution worker
- GRModelRunner: Model inference execution

Processing Components:
- GRProcessor: Input processor (GRPrompt -> GRRequest)
- GROutputProcessor: Output processor (predictions -> GRRequestOutput)

Configuration:
- GREngineArgs: Engine configuration arguments
"""

# Processors
from vllm_gr.engine.gr_processor import GRProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor, GRProcessedOutputs

# Engine arguments
from vllm_gr.engine.arg_utils import GREngineArgs

# Main engine
from vllm_gr.engine.gr_engine import GRLLMEngine

__all__ = [
    # Processors
    "GRProcessor",
    "GROutputProcessor",
    "GRProcessedOutputs",
    
    # Engine arguments
    "GREngineArgs",
    
    # Main engine
    "GRLLMEngine",
]
