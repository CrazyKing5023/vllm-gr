# vllm-GR

**vLLM-based Generative Recommender Inference Engine**

> 🚀 基础版本：支持 DLRM_v3 (HSTU) 模型的离线推理

vllm-GR 是基于 vLLM 生态构建的生成式推荐模型推理引擎。当前版本专注于 Meta DLRM_v3 模型的离线推理，该模型采用 HSTU (Hierarchical Sequential Transducer Unit) 架构。

## 功能特性

- ✅ **DLRM_v3 (HSTU) 模型** - 完整移植自 Meta generative-recommenders
- ✅ **离线推理** - 简洁的推荐生成 API
- ✅ **Top-K 采样** - 支持多任务预测和可配置的任务权重
- ✅ **灵活配置** - 易于使用的 JSON 配置格式
- ✅ **PyTorch 后端** - 纯 PyTorch 实现，无 Triton/CUDA 依赖

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-GR.git
cd vllm-GR

# Install dependencies
pip install -r requirements.txt

# Install vllm-GR
pip install -e .
```

### Basic Usage

```python
from vllm_gr import GRLLM
from vllm_gr.config.hstu_config import HSTUModelConfig

# Create configuration
config = HSTUModelConfig(
    num_layers=3,
    num_heads=4,
    embedding_dim=256,
    max_seq_len=512,
    max_num_candidates=100,
    task_names=["click", "like"],
)

# Initialize GRLLM
llm = GRLLM(config=config, device="cuda")

# Prepare request
user_id = "user_001"
sequence = [
    {"item_id": "movie_123", "timestamp": 1640995200},
    {"item_id": "movie_456", "timestamp": 1640995300},
    {"item_id": "movie_789", "timestamp": 1640995400},
]
candidates = [
    {"item_id": "movie_100"},
    {"item_id": "movie_101"},
    {"item_id": "movie_102"},
]

# Generate recommendations
response = llm.recommend(
    user_id=user_id,
    sequence=sequence,
    candidates=candidates,
    top_k=10,
)

# Access results
print(f"Latency: {response.latency_ms:.2f}ms")
for rec in response.recommendations:
    print(f"#{rec['rank']}: {rec['item_id']} (score: {rec['score']:.4f})")
```

### Loading Pretrained Models

```python
# Load from checkpoint directory
llm = GRLLM(model_path="./path/to/model", device="cuda")

# The checkpoint directory should contain:
# - config.json: Model configuration
# - model.pt or model.safetensors: Model weights
```

## Project Structure

```
vllm_gr/
├── __init__.py                 # Package exports
├── config/
│   └── hstu_config.py          # Model configuration
├── entrypoint/
│   └── gr.py                   # GRLLM main interface
├── engine/
│   ├── input_processor.py      # JSON → Model input
│   └── output_processor.py     # Model output → JSON
├── model_executor/
│   ├── embedding_manager.py    # Embedding management
│   ├── model_loader.py         # Model loading
│   └── models/
│       └── hstu.py             # HSTU model wrapper
├── models/
│   └── dlrm_v3/                # Ported DLRM_v3 code
│       ├── common.py
│       └── modules/
│           ├── dlrm_hstu.py
│           ├── stu.py
│           ├── hstu_transducer.py
│           └── ...
├── ops/                        # Operators
│   ├── hstu_attention.py
│   ├── hstu_compute.py
│   ├── jagged_tensors.py
│   └── pytorch/                # PyTorch implementations
│       ├── pt_hstu_attention.py
│       └── pt_jagged_tensors.py
└── sample/
    └── topk_sampler.py         # Top-K sampling
```

## 运行测试

```bash
# 运行所有测试
python tests/test_offline_inference.py

# 运行单个测试
python -c "from tests.test_offline_inference import test_grllm_basic; test_grllm_basic()"
```

## Examples

See the `examples/` directory for more usage examples:

- `simple_inference.py` - Basic single-user recommendation
- `batch_inference.py` - Batch recommendation for multiple users
- `config.json` - Example configuration file

```bash
# Run simple inference example
python examples/simple_inference.py

# Run batch inference example
python examples/batch_inference.py
```

## Creating Test Models

```bash
# Create a test model for development
python scripts/create_test_model.py --output-dir ./test_model

# Create with custom parameters
python scripts/create_test_model.py \
    --output-dir ./my_model \
    --num-layers 5 \
    --embedding-dim 512 \
    --hash-size 1000000
```

## API Reference

### GRLLM

```python
class GRLLM:
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[HSTUModelConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize GRLLM.
        
        Args:
            model_path: Path to pretrained model (optional)
            config: Model configuration (optional, loaded from model_path if not provided)
            device: Device to run on ("cuda" or "cpu")
        """
    
    def recommend(
        self,
        user_id: str,
        sequence: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        request_id: Optional[str] = None,
    ) -> RecommendationResponse:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            sequence: User interaction history
            candidates: Candidate items to rank
            top_k: Number of recommendations to return
            request_id: Optional request identifier
        
        Returns:
            RecommendationResponse with recommendations and latency
        """
    
    def recommend_batch(
        self,
        requests: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RecommendationResponse]:
        """
        Generate recommendations for multiple users.
        """
```

### RecommendationResponse

```python
@dataclass
class RecommendationResponse:
    request_id: str                        # Request identifier
    recommendations: List[Dict[str, Any]]  # Ranked recommendations
    latency_ms: float                      # Inference latency in milliseconds
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 5 | Number of HSTU layers |
| `num_heads` | 4 | Number of attention heads |
| `embedding_dim` | 512 | Transducer embedding dimension |
| `attention_dim` | 128 | Attention dimension |
| `hidden_dim` | 256 | Hidden dimension |
| `embedding_table_dim` | 192 | Embedding table dimension |
| `max_seq_len` | 1024 | Maximum sequence length |
| `max_num_candidates` | 100 | Maximum candidates per request |
| `hash_size` | 10,000,000 | Hash size for embeddings |
| `task_names` | ["is_click", "is_like"] | Multi-task names |

## 开发路线图

### 第一阶段：基础版本（当前）
- ✅ DLRM_v3 模型移植
- ✅ 离线推理
- ✅ Top-K 采样
- ✅ JSON 输入输出

### 第二阶段：性能优化
- ⬜ Triton 注意力内核
- ⬜ KV 缓存优化
- ⬜ 批处理优化

### 第三阶段：在线服务
- ⬜ GRScheduler 实现
- ⬜ API 服务器 (FastAPI)
- ⬜ CLI 工具
- ⬜ 连续批处理

### 第四阶段：生产化
- ⬜ 分布式嵌入
- ⬜ 监控指标
- ⬜ 压力测试

## License

This project is licensed under the Apache License 2.0. 

The DLRM_v3 model code is ported from Meta's [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) repository with modifications for vllm-GR.

## Acknowledgements

- [Meta's generative-recommenders](https://github.com/facebookresearch/generative-recommenders) - Original DLRM_v3 implementation
- [vLLM](https://github.com/vllm-project/vllm) - Inspiration for the inference engine architecture
