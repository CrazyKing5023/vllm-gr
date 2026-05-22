# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""FuXi-alpha inference model implementation."""

from vllm_gr.models.fuxi.fuxi_module import (
    FuXiAlphaInferenceModel,
    FuXiBlockJagged,
    FuXiJagged,
    MultistageFeedforwardNeuralNetwork,
    TIMESTAMPS_KEY,
    build_input_features_preprocessor,
    build_output_postprocessor,
)
from vllm_gr.models.fuxi.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor,
)
from vllm_gr.models.fuxi.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
    OutputPostprocessorModule,
)
from vllm_gr.models.fuxi.utils import get_current_embeddings

__all__ = [
    "FuXiAlphaInferenceModel",
    "FuXiBlockJagged",
    "FuXiJagged",
    "InputFeaturesPreprocessorModule",
    "LearnablePositionalEmbeddingInputFeaturesPreprocessor",
    "LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor",
    "L2NormEmbeddingPostprocessor",
    "LayerNormEmbeddingPostprocessor",
    "MultistageFeedforwardNeuralNetwork",
    "OutputPostprocessorModule",
    "TIMESTAMPS_KEY",
    "build_input_features_preprocessor",
    "build_output_postprocessor",
    "get_current_embeddings",
]
