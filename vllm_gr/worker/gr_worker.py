# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRWorker: Worker for GR model execution.

Manages GRModelRunner and handles model execution requests.
Inherits from vLLM's Worker to maintain API compatibility.
"""

from typing import Optional, Any, Dict
import logging

import torch

from vllm.v1.worker.gpu_worker import Worker

from vllm_gr.worker.gr_model_runner import GRModelRunner, GRModelRunnerOutput
from vllm_gr.core.gr_scheduler import GRSchedulerOutput

logger = logging.getLogger("vllm_gr.worker")


class GRWorker(Worker):
    """
    GR Worker - Inherits from vLLM Worker.
    
    Responsibilities:
    - Initialize device (GPU/CPU)
    - Create and manage GRModelRunner
    - Execute model inference
    
    Key differences from vLLM Worker:
    - No KV cache initialization (GR doesn't use KV cache)
    - Uses GRModelRunner instead of GPUModelRunner
    - Simplified for single-step recommendation inference
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        load_model: bool = True,
    ):
        """
        Initialize GRWorker.
        
        Args:
            model_path: Path to model weights
            device: Device string (cuda/cpu)
            embedding_dim: Embedding dimension
            hash_size: Hash size for embedding table
            max_seq_len: Maximum sequence length
            max_candidates: Maximum candidates
            num_layers: HSTU layers
            num_heads: Attention heads
            hidden_dim: Hidden dimension
            attention_dim: Attention dimension
            load_model: Whether to load model on init
        """
        # Set device - ensure proper format with index for CUDA
        if device is not None:
            # Normalize device string
            device_str = str(device).lower().strip()
            if device_str == "cuda" and torch.cuda.is_available():
                # Add default index if not specified
                device_str = "cuda:0"
            self._device = torch.device(device_str)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")
        
        self._model_path = model_path
        self._embedding_dim = embedding_dim
        self._hash_size = hash_size
        self._max_seq_len = max_seq_len
        self._max_candidates = max_candidates
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        self._attention_dim = attention_dim
        
        self.model_runner: Optional[GRModelRunner] = None
        self._initialized = False
        
        if load_model:
            self.init_device()
            self.load_model()
    
    def init_device(self) -> None:
        """Initialize the device."""
        if torch.cuda.is_available() and self._device.type == "cuda":
            # set_device expects an index or device with index
            device_index = self._device.index if self._device.index is not None else 0
            try:
                torch.cuda.set_device(device_index)
                # Verify CUDA is working
                torch.cuda.synchronize()
                logger.info(f"GRWorker CUDA device {device_index} initialized successfully")
            except RuntimeError as e:
                logger.warning(f"GRWorker CUDA init failed: {e}")
                # Try to recover
                torch.cuda.empty_cache()
                # Retry
                try:
                    torch.cuda.set_device(device_index)
                    logger.info(f"GRWorker CUDA device {device_index} recovered")
                except RuntimeError:
                    logger.warning("GRWorker falling back to CPU due to CUDA issues")
                    self._device = torch.device("cpu")
        logger.info(f"GRWorker device initialized: {self._device}")
    
    def load_model(self) -> None:
        """Load the model."""
        if self._initialized:
            return
        
        self.model_runner = GRModelRunner(
            model_path=self._model_path,
            device=self._device,
            embedding_dim=self._embedding_dim,
            hash_size=self._hash_size,
            max_seq_len=self._max_seq_len,
            max_candidates=self._max_candidates,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            hidden_dim=self._hidden_dim,
            attention_dim=self._attention_dim,
            load_model=True,
        )
        
        self._initialized = True
        logger.info("GRWorker model loaded")
    
    def execute_model(
        self,
        scheduler_output: GRSchedulerOutput,
    ) -> GRModelRunnerOutput:
        """
        Execute model inference.
        
        Args:
            scheduler_output: Output from scheduler
            
        Returns:
            Model runner output with predictions
        """
        if not self._initialized:
            self.load_model()
        
        return self.model_runner.execute_model(scheduler_output)
    
    def get_device(self) -> torch.device:
        """Get worker device."""
        return self._device
    
    @property
    def device(self) -> torch.device:
        """Device property."""
        return self._device
    
    def shutdown(self) -> None:
        """Shutdown worker and release resources."""
        if self.model_runner is not None:
            # Clear model from memory
            if hasattr(self.model_runner, 'model') and self.model_runner.model is not None:
                del self.model_runner.model
            self.model_runner = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("GRWorker shutdown complete")
