# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRModelRunner: Model runner for generative recommendation inference.

Handles model loading, embedding lookup, and inference execution.
"""

from typing import Any, Dict, Optional
import logging

import torch

logger = logging.getLogger("vllm_gr.model_runner")

from vllm_gr.inputs.data import GRInputs
from vllm_gr.request import GRRequest


class GRModelRunner:
    """
    Generative Recommendation Model Runner.
    
    Responsibilities:
    1. Load HSTUForRecommendation model
    2. Convert GRInputs (item IDs) to embeddings
    3. Execute model forward pass
    4. Return prediction scores
    
    Usage:
        # Load pretrained model
        runner = GRModelRunner(model_path="/path/to/model")
        
        # Create model from config (for testing)
        runner = GRModelRunner(num_layers=3, num_heads=2, ...)
        
        scores = runner.run_inference(gr_request, prompt)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        embedding_dim: int = 192,
        hash_size: int = 10_000_000,
        num_layers: int = 3,
        num_heads: int = 2,
        hidden_dim: int = 128,
        attention_dim: int = 64,
    ):
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.embedding_dim = embedding_dim
        self.hash_size = hash_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._model = None
        self._config = None
        self._initialized = False
        self._is_pretrained = False
        
        logger.info(f"GRModelRunner created: device={self.device}")
    
    def initialize(self) -> None:
        """Initialize model."""
        if self._initialized:
            return
        
        try:
            self._load_model()
            self._initialized = True
            mode = "pretrained" if self._is_pretrained else "config-based"
            logger.info(f"GRModelRunner initialized ({mode})")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise RuntimeError(f"Failed to create HSTU model: {e}") from e
    
    def _load_model(self) -> None:
        """Load or create HSTU model."""
        from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
        
        if self.model_path:
            # Load pretrained model
            from vllm_gr.model_executor.model_loader import GRModelLoader
            loader = GRModelLoader(model_path=self.model_path, device=self.device)
            self._model = loader.load()
            self._config = loader.config
            self._is_pretrained = True
            logger.info(f"Loaded pretrained model from {self.model_path}")
        else:
            # Create model from config
            logger.info("Creating HSTU model from config")
            
            self._config = HSTUModelConfig(
                max_seq_len=self.max_seq_len,
                max_num_candidates=self.max_candidates,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                attention_dim=self.attention_dim,
                embedding_dim=self.embedding_dim,
                embedding_table_dim=self.embedding_dim,
                hash_size=self.hash_size,
                embedding_tables=[
                    EmbeddingTableConfig(
                        name="item_embedding",
                        embedding_dim=self.embedding_dim,
                        num_embeddings=self.hash_size,
                        feature_names=["item_id", "item_candidate_id"],
                    ),
                ],
                user_embedding_feature_names=["item_id"],
                item_embedding_feature_names=["item_candidate_id"],
                uih_post_id_feature_name="item_id",
                is_inference=True,
            )
            
            self._model = HSTUForRecommendation.from_model_config(
                config=self._config,
                device=self.device,
            )
            self._model.to(self.device)
            self._model.eval()
            self._is_pretrained = False
    
    @property
    def is_pretrained(self) -> bool:
        return self._is_pretrained
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def model(self):
        return self._model
    
    @property
    def config(self):
        return self._config
    
    def run_inference(
        self,
        gr_request: GRRequest,
        prompt: Dict[str, Any],
    ) -> torch.Tensor:
        """Execute inference."""
        if not self._initialized:
            self.initialize()
        
        # Extract item IDs
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        
        uih_item_ids = [item.get("item_id", 0) for item in user_history]
        cand_item_ids = [item.get("item_id", 0) for item in candidates]
        
        # Limit length
        uih_item_ids = uih_item_ids[-self.max_seq_len:]
        cand_item_ids = cand_item_ids[:self.max_candidates]
        
        num_uih = len(uih_item_ids)
        num_cand = len(cand_item_ids)
        
        if num_cand == 0:
            return torch.tensor([], device=self.device)
        
        # Convert to tensors
        uih_ids_tensor = torch.tensor(uih_item_ids, dtype=torch.long, device=self.device)
        cand_ids_tensor = torch.tensor(cand_item_ids, dtype=torch.long, device=self.device)
        
        # Lookup embeddings using model's embedding tables
        if num_uih > 0:
            uih_embeddings = self._model.lookup_embeddings(uih_ids_tensor)
        else:
            uih_embeddings = torch.zeros(0, self.embedding_dim, device=self.device)
        
        cand_embeddings = self._model.lookup_embeddings(cand_ids_tensor)
        
        # Length tensors
        uih_lengths = torch.tensor([num_uih], dtype=torch.long, device=self.device)
        num_candidates = torch.tensor([num_cand], dtype=torch.long, device=self.device)
        
        # Run inference
        with torch.no_grad():
            output = self._model.run_inference(
                uih_embeddings=uih_embeddings,
                uih_lengths=uih_lengths,
                candidate_embeddings=cand_embeddings,
                num_candidates=num_candidates,
            )
        
        # Extract scores
        predictions = output.predictions
        if predictions is None:
            raise RuntimeError("Model returned no predictions")
        
        if predictions.dim() > 1:
            scores = predictions[0]
        else:
            scores = predictions
        
        return scores
    
    def shutdown(self) -> None:
        """Shutdown and release resources."""
        self._model = None
        self._config = None
        self._initialized = False
        self._is_pretrained = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GRModelRunner shutdown")
