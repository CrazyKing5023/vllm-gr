# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Model Loader for vllm-GR.

Loads HSTU models from PyTorch checkpoint or SafeTensors format.
"""

import os
import logging
from typing import Tuple, Optional, Dict

import torch

from vllm_gr.config.hstu_config import HSTUModelConfig
from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

logger = logging.getLogger(__name__)


class GRModelLoader:
    """
    HSTU Model Loader.
    
    Supported formats:
    - PyTorch checkpoint (.pt, .pth)
    - SafeTensors (.safetensors)
    - Directory containing config.json and model files
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[HSTUModelConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config()
    
    def _load_config(self) -> HSTUModelConfig:
        """Load configuration from model path."""
        if os.path.isdir(self.model_path):
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                return HSTUModelConfig.from_pretrained(self.model_path)
        return HSTUModelConfig()
    
    def load(self) -> HSTUForRecommendation:
        """
        Load model.
        
        Returns:
            HSTUForRecommendation model
        """
        logger.info(f"Loading model from {self.model_path}")
        
        # Create model
        model = HSTUForRecommendation.from_model_config(
            config=self.config,
            device=self.device,
        )
        
        # Try to load weights
        weights_loaded = self._load_weights(model)
        if weights_loaded:
            logger.info("Model weights loaded successfully")
        else:
            logger.warning("No weights found, using randomly initialized model")
        
        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_weights(self, model: HSTUForRecommendation) -> bool:
        """Load weights from checkpoint."""
        weight_files = []
        
        if os.path.isfile(self.model_path):
            weight_files = [self.model_path]
        elif os.path.isdir(self.model_path):
            for ext in [".pt", ".pth", ".bin", ".safetensors"]:
                for filename in os.listdir(self.model_path):
                    if filename.endswith(ext):
                        weight_files.append(os.path.join(self.model_path, filename))
        
        if not weight_files:
            return False
        
        for weight_file in weight_files:
            try:
                state_dict = self._load_state_dict(weight_file)
                if state_dict:
                    self._apply_weights(model, state_dict)
                    return True
            except Exception as e:
                logger.warning(f"Failed to load weights from {weight_file}: {e}")
        
        return False
    
    def _load_state_dict(self, path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load state dict from file."""
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                return load_file(path)
            except ImportError:
                logger.warning("safetensors not installed")
                return None
        else:
            return torch.load(path, map_location=self.device)
    
    def _apply_weights(
        self,
        model: HSTUForRecommendation,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Apply loaded weights to model."""
        # Use vLLM-style weight loading
        weights_iter = ((k, v) for k, v in state_dict.items())
        model.load_weights(weights_iter)
