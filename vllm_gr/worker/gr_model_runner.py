# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner: Model runner for HSTU recommendation inference.

Handles model loading and batch inference for GR requests.
Inherits from vLLM's GPUModelRunner to maintain API compatibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging

import torch
import torch.nn as nn

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.core.gr_scheduler import GRSchedulerOutput, ScheduledGRRequest

logger = logging.getLogger("vllm_gr.model_runner")


@dataclass
class GRModelRunnerOutput:
    """Output from GRModelRunner.execute_model()."""
    
    request_ids: List[str]
    """Request IDs in order."""
    
    predictions: List[torch.Tensor]
    """Prediction scores per request."""
    
    user_embeddings: Optional[List[torch.Tensor]] = None
    """User embeddings per request (optional)."""


class GRModelRunner(GPUModelRunner):
    """
    GR Model Runner - Inherits from vLLM GPUModelRunner.
    
    Responsibilities:
    - Load HSTUForRecommendation model
    - Execute batched inference
    - Return prediction scores per request
    
    Key differences from vLLM GPUModelRunner:
    - No KV cache management
    - Single-step recommendation inference (non-autoregressive)
    - Works with item IDs instead of token IDs
    - Returns prediction scores instead of logits
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        # HSTU architecture parameters
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        load_model: bool = True,
    ):
        """
        Initialize GRModelRunner.
        
        Args:
            model_path: Path to pretrained model (optional)
            device: Device for inference
            embedding_dim: Embedding vector dimension
            hash_size: Embedding table hash size
            max_seq_len: Maximum sequence length
            max_candidates: Maximum candidates
            num_layers: Number of HSTU layers
            num_heads: Number of attention heads
            hidden_dim: Hidden layer dimension
            attention_dim: Attention dimension
            load_model: Whether to load the model on init
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.hash_size = hash_size
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Set device - ensure proper format with index for CUDA
        if device is not None:
            if isinstance(device, str):
                device_str = device.lower().strip()
                if device_str == "cuda" and torch.cuda.is_available():
                    device_str = "cuda:0"
                self._device = torch.device(device_str)
            else:
                self._device = device
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")
        
        self.model: Optional[nn.Module] = None
        self._initialized = False
        
        if load_model:
            self.load_model()
    
    def load_model(self) -> None:
        """Load the HSTU model."""
        if self._initialized:
            return
        
        import os
        import json
        
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
        from vllm_gr.config.hstu_config import HSTUModelConfig
        
        # Initialize CUDA device if needed
        if self._device.type == "cuda":
            device_index = self._device.index if self._device.index is not None else 0
            try:
                torch.cuda.set_device(device_index)
                # Clear any previous CUDA errors
                torch.cuda.synchronize()
            except RuntimeError as e:
                logger.warning(f"CUDA initialization warning: {e}")
                # Reset CUDA context if there was an error
                torch.cuda.empty_cache()
        
        # Check if model_path is a directory with config.json
        config_loaded_from_path = False
        if self.model_path and os.path.isdir(self.model_path):
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    # Override parameters from config.json
                    self.num_layers = config_dict.get("num_layers", self.num_layers)
                    self.num_heads = config_dict.get("num_heads", self.num_heads)
                    self.embedding_dim = config_dict.get("embedding_dim", self.embedding_dim)
                    self.attention_dim = config_dict.get("attention_dim", self.attention_dim)
                    self.hidden_dim = config_dict.get("hidden_dim", self.hidden_dim)
                    self.max_seq_len = config_dict.get("max_seq_len", self.max_seq_len)
                    self.max_candidates = config_dict.get("max_num_candidates", self.max_candidates)
                    self.hash_size = config_dict.get("hash_size", self.hash_size)
                    config_loaded_from_path = True
                    logger.info(f"Loaded model config from {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Create model config
        model_config = HSTUModelConfig(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            attention_dim=self.attention_dim,
            hidden_dim=self.hidden_dim,
            embedding_table_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            max_num_candidates=self.max_candidates,
            hash_size=self.hash_size,
        )
        
        # Create model on CPU first, then move to device
        logger.info(f"Creating HSTUForRecommendation model...")
        try:
            self.model = HSTUForRecommendation.from_model_config(
                config=model_config,
                device=torch.device("cpu"),  # Create on CPU first
            )
            logger.info(f"Model created successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
        
        # Load weights if path provided and weights file exists
        weights_loaded = False
        if self.model_path:
            weights_loaded = self._load_weights(self.model_path)
        
        # Move to device and eval mode
        logger.info(f"Moving model to {self._device}...")
        try:
            # Move to device in stages to identify any issues
            self.model.to(self._device)
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            logger.info(f"Model moved to {self._device}")
        except RuntimeError as e:
            logger.error(f"Failed to move model to {self._device}: {e}")
            # Try to recover CUDA state
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        
        self.model.eval()
        
        self._initialized = True
        
        # Log initialization status
        if config_loaded_from_path and not weights_loaded:
            logger.info(
                f"GRModelRunner initialized on {self._device} "
                f"(config from {self.model_path}, randomly initialized weights)"
            )
        elif weights_loaded:
            logger.info(f"GRModelRunner initialized on {self._device} (weights loaded)")
        else:
            logger.info(f"GRModelRunner initialized on {self._device} (randomly initialized)")
    
    def _load_weights(self, model_path: str) -> bool:
        """
        Load model weights from path.
        
        Args:
            model_path: Path to model directory or weights file
            
        Returns:
            True if weights were loaded successfully, False otherwise
        """
        import os
        
        # Determine weights file path
        if os.path.isdir(model_path):
            # Check for common weight file names in directory
            weight_files = [
                "pytorch_model.bin",
                "model.safetensors",
                "model.pt",
                "weights.pt",
            ]
            weights_path = None
            for wf in weight_files:
                candidate = os.path.join(model_path, wf)
                if os.path.exists(candidate):
                    weights_path = candidate
                    break
            
            if weights_path is None:
                logger.info(
                    f"No weights file found in {model_path}, "
                    "using randomly initialized weights (demo mode)"
                )
                return False
        else:
            weights_path = model_path
        
        if not os.path.exists(weights_path):
            logger.info(f"Weights path {weights_path} does not exist, using random weights")
            return False
        
        try:
            state_dict = torch.load(weights_path, map_location=self._device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}, using random weights")
            return False
    
    def execute_model(
        self,
        scheduler_output: GRSchedulerOutput,
    ) -> GRModelRunnerOutput:
        """
        Execute model inference on scheduled requests.
        
        Args:
            scheduler_output: Output from GRScheduler.schedule()
            
        Returns:
            GRModelRunnerOutput with predictions per request
        """
        if not self._initialized:
            self.load_model()
        
        if scheduler_output.is_empty:
            return GRModelRunnerOutput(
                request_ids=[],
                predictions=[],
            )
        
        requests = scheduler_output.scheduled_requests
        
        # Collect all item IDs
        all_uih_ids: List[int] = []
        all_cand_ids: List[int] = []
        uih_lengths: List[int] = []
        num_candidates: List[int] = []
        request_ids: List[str] = []
        
        for req in requests:
            request_ids.append(req.request_id)
            
            # User history
            uih_ids = req.prompt_token_ids
            all_uih_ids.extend(uih_ids)
            uih_lengths.append(len(uih_ids))
            
            # Candidates
            cand_ids = req.candidate_item_ids
            all_cand_ids.extend(cand_ids)
            num_candidates.append(len(cand_ids))
        
        # Prepare tensors
        uih_item_ids = torch.tensor(all_uih_ids, dtype=torch.long, device=self._device)
        candidate_item_ids = torch.tensor(all_cand_ids, dtype=torch.long, device=self._device)
        uih_lengths_tensor = torch.tensor(uih_lengths, dtype=torch.long, device=self._device)
        num_candidates_tensor = torch.tensor(num_candidates, dtype=torch.long, device=self._device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.run_recommendation(
                uih_item_ids=uih_item_ids,
                uih_lengths=uih_lengths_tensor,
                candidate_item_ids=candidate_item_ids,
                num_candidates=num_candidates_tensor,
            )
        
        # Split predictions per request
        prediction_list = self._split_predictions(predictions, num_candidates)
        
        return GRModelRunnerOutput(
            request_ids=request_ids,
            predictions=prediction_list,
        )
    
    def _split_predictions(
        self,
        predictions: torch.Tensor,
        num_candidates: List[int],
    ) -> List[torch.Tensor]:
        """Split batch predictions into per-request predictions."""
        prediction_list = []
        
        # predictions shape: (num_tasks, total_candidates) or (total_candidates,)
        if predictions.dim() == 1:
            scores = predictions
        else:
            # Use first task (usually main prediction)
            scores = predictions[0]
        
        offset = 0
        for n_cand in num_candidates:
            req_scores = scores[offset:offset + n_cand].clone()
            prediction_list.append(req_scores)
            offset += n_cand
        
        return prediction_list
    
    def get_device(self) -> torch.device:
        """Get the device for this runner."""
        return self._device
    
    @property
    def device(self) -> torch.device:
        """Device property."""
        return self._device
