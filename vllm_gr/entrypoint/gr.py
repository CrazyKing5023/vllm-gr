# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLM Offline Inference Interface

Provides a concise recommendation generation API supporting single and batch request inference.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import logging

import torch

from vllm_gr.config.hstu_config import HSTUModelConfig
from vllm_gr.model_executor.model_loader import GRModelLoader
from vllm_gr.model_executor.embedding_manager import EmbeddingManager
from vllm_gr.engine.input_processor import GRInputProcessor
from vllm_gr.engine.output_processor import GROutputProcessor
from vllm_gr.sample.topk_sampler import TopKSampler


logger = logging.getLogger(__name__)


@dataclass
class RecommendationResponse:
    """Recommendation response"""
    request_id: str
    recommendations: List[Dict[str, Any]]
    latency_ms: float


class GRLLM:
    """
    Generative Recommender Offline Inference Interface
    
    Usage example:
        llm = GRLLM(model_path="path/to/hstu-model")
        response = llm.recommend(
            user_id="user_001",
            sequence=[{"item_id": "item_1", "timestamp": 1234567890}, ...],
            candidates=[{"item_id": "item_100"}, {"item_id": "item_101"}, ...],
            top_k=10,
        )
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[HSTUModelConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize GRLLM
        
        Args:
            model_path: Model path (optional, uses default config if not provided)
            config: Model configuration (optional)
            device: Runtime device ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. Load or create configuration
        if config is not None:
            self.config = config
        elif model_path is not None:
            self.config = HSTUModelConfig.from_pretrained(model_path)
        else:
            self.config = HSTUModelConfig()
        
        # 2. Load model
        if model_path is not None:
            loader = GRModelLoader(model_path, self.config, self.device)
            self.model, self.embedding_manager = loader.load()
        else:
            # Create default model and embedding manager
            self.embedding_manager = EmbeddingManager(self.config, self.device)
            from vllm_gr.model_executor.models.hstu import HSTUModelWrapper
            self.model = HSTUModelWrapper(
                config=self.config,
                embedding_manager=self.embedding_manager,
                device=self.device,
            )
            self.model.to(self.device)
        
        self.model.eval()
        
        # 3. Initialize processors
        self.input_processor = GRInputProcessor(self.config)
        self.output_processor = GROutputProcessor(self.config)
        self.sampler = TopKSampler()
        
        logger.info(f"GRLLM initialized on {self.device}")
    
    def recommend(
        self,
        user_id: str,
        sequence: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        request_id: Optional[str] = None,
    ) -> RecommendationResponse:
        """
        Generate recommendations
        
        Args:
            user_id: User ID
            sequence: User behavior sequence [{"item_id": ..., "timestamp": ..., ...}, ...]
            candidates: Candidate items [{"item_id": ...}, ...]
            top_k: Number of recommendations to return
            request_id: Request ID (optional)
        
        Returns:
            RecommendationResponse: Recommendation results
        """
        start_time = time.time()
        
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        
        with torch.no_grad():
            # 1. Input processing: convert to indices
            (
                sequence_indices,
                sequence_lengths,
                candidate_indices,
                num_candidates,
            ) = self.input_processor.process(
                user_id=user_id,
                sequence=sequence,
                candidates=candidates,
                device=self.device,
            )
            
            # 2. Embedding lookup
            sequence_embeddings = self.embedding_manager.lookup("item_id", sequence_indices)
            candidate_embeddings = self.embedding_manager.lookup("item_id", candidate_indices)
            
            # 3. Model inference
            model_output = self.model(
                uih_embeddings=sequence_embeddings,
                uih_lengths=sequence_lengths,
                candidate_embeddings=candidate_embeddings,
                num_candidates=num_candidates,
            )
            
            # 4. Top-K sampling
            candidate_ids = [c["item_id"] for c in candidates]
            sampled = self.sampler.sample(
                predictions=model_output.predictions,
                candidate_ids=candidate_ids,
                top_k=top_k,
            )
            
            # 5. Output processing
            recommendations = self.output_processor.format_output(
                sampled=sampled,
                task_names=self.config.task_names,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            request_id=request_id,
            recommendations=recommendations,
            latency_ms=latency_ms,
        )
    
    def recommend_batch(
        self,
        requests: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RecommendationResponse]:
        """
        Generate batch recommendations
        
        Args:
            requests: Request list, each request contains user_id, sequence, candidates
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendation responses
        """
        responses = []
        for i, req in enumerate(requests):
            response = self.recommend(
                user_id=req.get("user_id", f"user_{i}"),
                sequence=req.get("sequence", []),
                candidates=req.get("candidates", []),
                top_k=top_k,
                request_id=req.get("request_id"),
            )
            responses.append(response)
        return responses
