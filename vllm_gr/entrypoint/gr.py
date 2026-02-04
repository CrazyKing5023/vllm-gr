# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
GRLLM 离线推理接口

提供简洁的推荐生成 API，支持单请求和批量请求推理。
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
    """推荐响应"""
    request_id: str
    recommendations: List[Dict[str, Any]]
    latency_ms: float


class GRLLM:
    """
    生成式推荐离线推理接口
    
    使用示例:
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
        初始化 GRLLM
        
        Args:
            model_path: 模型路径（可选，如果不提供则使用默认配置）
            config: 模型配置（可选）
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 加载或创建配置
        if config is not None:
            self.config = config
        elif model_path is not None:
            self.config = HSTUModelConfig.from_pretrained(model_path)
        else:
            self.config = HSTUModelConfig()
        
        # 2. 加载模型
        if model_path is not None:
            loader = GRModelLoader(model_path, self.config, self.device)
            self.model, self.embedding_manager = loader.load()
        else:
            # 创建默认模型和嵌入管理器
            self.embedding_manager = EmbeddingManager(self.config, self.device)
            from vllm_gr.model_executor.models.hstu import HSTUModelWrapper
            self.model = HSTUModelWrapper(
                config=self.config,
                embedding_manager=self.embedding_manager,
                device=self.device,
            )
            self.model.to(self.device)
        
        self.model.eval()
        
        # 3. 初始化处理器
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
        生成推荐
        
        Args:
            user_id: 用户 ID
            sequence: 用户行为序列 [{"item_id": ..., "timestamp": ..., ...}, ...]
            candidates: 候选物品 [{"item_id": ...}, ...]
            top_k: 返回的推荐数量
            request_id: 请求 ID（可选）
        
        Returns:
            RecommendationResponse: 推荐结果
        """
        start_time = time.time()
        
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        
        with torch.no_grad():
            # 1. 输入处理：转换为索引
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
            
            # 2. 嵌入查找
            sequence_embeddings = self.embedding_manager.lookup("item_id", sequence_indices)
            candidate_embeddings = self.embedding_manager.lookup("item_id", candidate_indices)
            
            # 3. 模型推理
            model_output = self.model(
                uih_embeddings=sequence_embeddings,
                uih_lengths=sequence_lengths,
                candidate_embeddings=candidate_embeddings,
                num_candidates=num_candidates,
            )
            
            # 4. Top-K 采样
            candidate_ids = [c["item_id"] for c in candidates]
            sampled = self.sampler.sample(
                predictions=model_output.predictions,
                candidate_ids=candidate_ids,
                top_k=top_k,
            )
            
            # 5. 输出处理
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
        批量生成推荐
        
        Args:
            requests: 请求列表，每个请求包含 user_id, sequence, candidates
            top_k: 返回的推荐数量
        
        Returns:
            推荐响应列表
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

