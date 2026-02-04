# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
模型加载器

支持从 PyTorch checkpoint 或 SafeTensors 格式加载 HSTU 模型。
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any

import torch

from vllm_gr.config.hstu_config import HSTUModelConfig
from vllm_gr.model_executor.embedding_manager import EmbeddingManager
from vllm_gr.model_executor.models.hstu import HSTUModelWrapper

logger = logging.getLogger(__name__)


class GRModelLoader:
    """
    HSTU 模型加载器
    
    支持的加载格式：
    - PyTorch checkpoint (.pt, .pth)
    - SafeTensors 格式 (.safetensors)
    - 包含 config.json 和模型文件的目录
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[HSTUModelConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load or use provided config
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
        
        # Return default config
        return HSTUModelConfig()
    
    def load(self) -> Tuple[HSTUModelWrapper, EmbeddingManager]:
        """
        Load model and embedding manager.
        
        Returns:
            (model, embedding_manager)
        """
        logger.info(f"Loading model from {self.model_path}")
        
        # Create embedding manager
        embedding_manager = EmbeddingManager(self.config, self.device)
        
        # Create model wrapper
        model = HSTUModelWrapper(
            config=self.config,
            embedding_manager=embedding_manager,
            device=self.device,
        )
        
        # Try to load weights
        weights_loaded = self._load_weights(model, embedding_manager)
        if weights_loaded:
            logger.info("Model weights loaded successfully")
        else:
            logger.warning("No weights found, using randomly initialized model")
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        return model, embedding_manager
    
    def _load_weights(
        self,
        model: HSTUModelWrapper,
        embedding_manager: EmbeddingManager,
    ) -> bool:
        """
        Load weights from checkpoint.
        
        Returns:
            True if weights were loaded, False otherwise
        """
        weight_files = []
        
        if os.path.isfile(self.model_path):
            weight_files = [self.model_path]
        elif os.path.isdir(self.model_path):
            # Look for weight files
            for ext in [".pt", ".pth", ".bin", ".safetensors"]:
                for filename in os.listdir(self.model_path):
                    if filename.endswith(ext):
                        weight_files.append(os.path.join(self.model_path, filename))
        
        if not weight_files:
            return False
        
        # Load the first valid weight file
        for weight_file in weight_files:
            try:
                state_dict = self._load_state_dict(weight_file)
                if state_dict:
                    self._apply_weights(model, embedding_manager, state_dict)
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
                logger.warning("safetensors not installed, skipping .safetensors file")
                return None
        else:
            return torch.load(path, map_location=self.device)
    
    def _apply_weights(
        self,
        model: HSTUModelWrapper,
        embedding_manager: EmbeddingManager,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Apply loaded weights to model and embedding manager."""
        # Separate embedding weights from model weights
        embedding_weights = {}
        model_weights = {}
        
        for key, value in state_dict.items():
            if "embedding" in key.lower():
                embedding_weights[key] = value
            else:
                model_weights[key] = value
        
        # Load embedding weights
        if embedding_weights:
            embedding_manager.load_weights(embedding_weights)
        
        # Load model weights (with flexible key matching)
        if model_weights:
            try:
                model.load_state_dict(model_weights, strict=False)
            except Exception as e:
                logger.warning(f"Partial model weight loading: {e}")

