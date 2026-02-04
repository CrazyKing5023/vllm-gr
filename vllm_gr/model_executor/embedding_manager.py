# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
嵌入管理器

管理特征嵌入表的创建、查找和权重加载。
当前版本仅支持单 GPU，不支持分布式分片。
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig


class SimpleEmbeddingTable(nn.Module):
    """简单嵌入表实现，基于 PyTorch nn.Embedding。"""
    
    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embedding_dim: int,
        feature_names: List[str],
    ):
        super().__init__()
        self.name = name
        self.feature_names = feature_names
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize with small values
        nn.init.uniform_(self.embedding.weight, -0.01, 0.01)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings by indices."""
        return self.embedding(indices)


class EmbeddingManager:
    """
    嵌入管理器
    
    管理多个嵌入表，支持按特征名称进行嵌入查找。
    
    当前限制：
    - 仅支持单 GPU
    - 不支持量化
    - 不支持分布式分片
    """
    
    def __init__(
        self,
        config: HSTUModelConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        
        # Create embedding tables
        self.embedding_tables: Dict[str, SimpleEmbeddingTable] = {}
        self.feature_to_table: Dict[str, str] = {}
        
        self._create_embedding_tables()
    
    def _create_embedding_tables(self) -> None:
        """Create embedding tables based on config."""
        for table_config in self.config.embedding_tables:
            table = SimpleEmbeddingTable(
                name=table_config.name,
                num_embeddings=table_config.num_embeddings,
                embedding_dim=table_config.embedding_dim,
                feature_names=table_config.feature_names,
            ).to(self.device)
            
            self.embedding_tables[table_config.name] = table
            
            # Map features to tables
            for feature_name in table_config.feature_names:
                self.feature_to_table[feature_name] = table_config.name
    
    def lookup(
        self,
        feature_name: str,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up embeddings for a feature.
        
        Args:
            feature_name: Name of the feature
            indices: Tensor of indices (L,) or (B, L)
        
        Returns:
            Embedding tensor (L, D) or (B, L, D)
        """
        if feature_name not in self.feature_to_table:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        table_name = self.feature_to_table[feature_name]
        table = self.embedding_tables[table_name]
        
        # Ensure indices are on the right device and within bounds
        indices = indices.to(self.device)
        indices = indices.clamp(0, table.embedding.num_embeddings - 1)
        
        return table(indices)
    
    def lookup_batch(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Look up embeddings for multiple features.
        
        Args:
            features: Dict mapping feature names to index tensors
        
        Returns:
            Dict mapping feature names to embedding tensors
        """
        embeddings = {}
        for feature_name, indices in features.items():
            if feature_name in self.feature_to_table:
                embeddings[feature_name] = self.lookup(feature_name, indices)
        return embeddings
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load embedding weights from state dict.
        
        Args:
            state_dict: Dictionary containing embedding weights
        """
        for table_name, table in self.embedding_tables.items():
            weight_key = f"{table_name}.weight"
            embedding_key = f"embedding_tables.{table_name}.embedding.weight"
            
            if weight_key in state_dict:
                table.embedding.weight.data.copy_(state_dict[weight_key])
            elif embedding_key in state_dict:
                table.embedding.weight.data.copy_(state_dict[embedding_key])
    
    def parameters(self):
        """Return all embedding parameters."""
        for table in self.embedding_tables.values():
            yield from table.parameters()
    
    def to(self, device: torch.device) -> "EmbeddingManager":
        """Move all tables to device."""
        self.device = device
        for table in self.embedding_tables.values():
            table.to(device)
        return self
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self.embedding_tables:
            first_table = next(iter(self.embedding_tables.values()))
            return first_table.embedding.embedding_dim
        return self.config.embedding_tables[0].embedding_dim if self.config.embedding_tables else 512

