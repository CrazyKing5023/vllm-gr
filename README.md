# vllm-GR

**vLLM-based Generative Recommender Inference Engine**

> 🚀 Foundation Version: Supports offline inference for DLRM_v3 (HSTU) model

vllm-GR is a generative recommendation model inference engine built on the vLLM ecosystem. The current version focuses on offline inference for Meta's DLRM_v3 model, which uses the HSTU (Hierarchical Sequential Transducer Unit) architecture.

## Features

- ✅ **DLRM_v3 (HSTU) Model** - Fully ported from Meta generative-recommenders
- ✅ **Offline Inference** - Simple recommendation generation API
- ✅ **Top-K Sampling** - Supports multi-task prediction with configurable task weights
- ✅ **Flexible Configuration** - Easy-to-use JSON configuration format
- ✅ **PyTorch Backend** - Pure PyTorch implementation, no Triton/CUDA dependencies

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

## Running Tests

```bash
# Run all tests
python tests/test_offline_inference.py

# Run individual test
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

## Development Roadmap

### Phase 1: Foundation Version (Current)
- ✅ DLRM_v3 model porting
- ✅ Offline inference
- ✅ Top-K sampling
- ✅ JSON input/output

### Phase 2: Online Service
- ⬜ GRScheduler implementation
- ⬜ API server (FastAPI)
- ⬜ CLI tools
- ⬜ Continuous batching

### Phase 3: Performance Optimization
- ⬜ Triton attention kernels
- ⬜ KV cache optimization
- ⬜ Batch processing optimization

### Phase 4: Production-Ready
- ⬜ Distributed embeddings
- ⬜ Monitoring metrics
- ⬜ Stress testing

## License

This project is licensed under the Apache License 2.0. 

The DLRM_v3 model code is ported from Meta's [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) repository with modifications for vllm-GR.

## Acknowledgements

- [Meta's generative-recommenders](https://github.com/facebookresearch/generative-recommenders) - Original DLRM_v3 implementation
- [vLLM](https://github.com/vllm-project/vllm) - Inspiration for the inference engine architecture
