# vllm-GR

**vLLM-based Generative Recommender Inference Engine**

> Foundation Version: Supports offline inference for DLRM_v3 (HSTU) model

vllm-GR is a generative recommendation model inference engine built on the vLLM ecosystem. The current version focuses on offline inference for Meta's DLRM_v3 model, which uses the HSTU (Hierarchical Sequential Transducer Unit) architecture.

## Features

- **GRLLM Interface** - Inherits from vLLM LLM class, provides recommendation-specific API
- **GRPrompt Input** - Structured input format for recommendations
- **GRRequestOutput** - Type-safe output with RecommendedItem objects
- **Batch Processing** - Efficient batch recommendation support
- **Full vLLM Integration** - Direct inheritance from vLLM classes
- **跨平台运行时** - 共享 GR/HSTU 逻辑，并在 GPU / NPU 适配边界处分层

## 平台治理

GPU / NPU 共存的仓库治理规则见 [PLATFORM_GOVERNANCE.md](PLATFORM_GOVERNANCE.md)。
当前实现的分层与调用链见 [ARCHITECTURE.md](ARCHITECTURE.md)。
主线功能合入时的执行清单见 [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)。

简要原则如下：

- 保持一个主线仓库和一套共享业务逻辑
- 平台专用代码只允许出现在 `worker/`、`utils/`、以及面向后端的 `ops/` 边界层
- 不长期维护重复的 GPU / NPU 业务分支

## Dependencies

### Shared Layer

- `msgspec`
- `tqdm>=4.66.0`

### GPU Runtime

- `torch==2.9.1`
- `numpy==2.2.6`
- `opencv-python-headless>=4.13.0`
- `vllm==0.14.1`
- `torchrec==1.4.0`
- `fbgemm-gpu==1.4.0`

### NPU Runtime

- `opencv-python-headless>=4.13.0`
- `vllm==0.14.1`
- `fbgemm-gpu-cpu==1.4.0`
- `torch==2.9.1` 需由 NPU 基础环境预装
- `torch-npu==2.9.0` 需由 NPU 基础环境预装
- `numpy` 建议跟随 NPU 基础环境 / 上游栈预装，不由本仓直接安装
- `vllm_ascend==0.14.0rc1` 需由 NPU 基础环境预装
- `torchrec==1.4.0+npu` 由 RecSDK 单独安装

NPU 运行说明：

- 本仓仍然遵循“同仓共存，但单次运行只选择 GPU 或 NPU 其中一条路径”
- 当前仓的 NPU 基线固定为 `torch==2.9.1`、`torch-npu==2.9.0`、`vllm_ascend==0.14.0rc1`
- `opencv-python-headless` 与 GPU 路径保持一致，当前口径为 `>=4.13.0`
- `torch==2.9.1` 不由本仓依赖直接安装，要求随 Ascend / CANN 环境预先就绪
- `torch-npu==2.9.0` 不由本仓依赖直接安装，要求随 Ascend / CANN 环境预先就绪
- `numpy` 不由本仓依赖直接安装，建议保持与 NPU 基础环境一致
- `vllm_ascend==0.14.0rc1` 不由本仓依赖直接安装，要求随 Ascend / CANN 环境预先就绪
- `torchrec==1.4.0+npu` 不由本仓直接安装，要求先完成 RecSDK 环境安装
- NPU 平台初始化继续委托给 `vllm_ascend`

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-GR.git
cd vllm-GR

# Shared layer only
pip install .

# Or: shared layer from requirements shim
pip install -r requirements.txt

# Development install (editable mode)
pip install -e ".[dev]"

# GPU runtime
pip install -r requirements-gpu.txt
# or
pip install ".[gpu]"

# Full GPU runtime + dev + serving extras
pip install ".[all-gpu]"

# NPU runtime
# torch==2.9.1 should already exist in the base NPU environment
# torch-npu==2.9.0 should already exist in the base NPU environment
# numpy should also follow the base NPU environment if it is pinned there
# vllm_ascend==0.14.0rc1 should already exist in the base NPU environment
pip install -r requirements-npu.txt
# or
pip install ".[npu]"

# Full NPU runtime + dev + serving extras
pip install ".[all-npu]"
```

安装建议：

- 新环境优先使用 `requirements-gpu.txt` 或 `requirements-npu.txt`
- `.[all]` 目前只保留为兼容别名，默认对齐 GPU 路径；新流程不建议继续把它当作“双平台通用安装”
- 如果要走 NPU 路径，优先使用预装 `torch==2.9.1`、预装 `torch-npu==2.9.0`、预装 `vllm_ascend==0.14.0rc1` 这一组固定版本
- 如果要走 NPU 路径，需额外确认 RecSDK 已提供 `torchrec==1.4.0+npu`

### Final NPU Install Order

建议固定使用下面的顺序：

1. 基础环境
   预装 `torch==2.9.1`、`torch-npu==2.9.0`、`numpy`、`vllm_ascend==0.14.0rc1`，并确认 `torch.npu.is_available()` 为 `True`。
2. RecSDK
   单独安装并确认可提供 `torchrec==1.4.0+npu`。
3. GR 主仓
   进入仓库后执行 `pip install -r requirements-npu.txt`；如果要跑测试或验收，执行 `pip install -e ".[all-npu]"`。

建议补一个最小确认：

```bash
cd /root/PROJECT/HSTU_Update/GR_Repo_main

python tests/test_npu_smoke.py
python scripts/run_tests.py --suite platform-npu
```

### Real Model Smoke

建议把真实模型 smoke 固定为后续回归入口：

```bash
cd /root/PROJECT/HSTU_Update/GR_Repo_main

MODEL_DIR=/path/to/converted_model

python examples/load_converted_model.py \
    --model "$MODEL_DIR"
```

如果训练 checkpoint 还未转换，可先执行：

```bash
cd /root/PROJECT/HSTU_Update/GR_Repo_main

CKPT_DIR=/path/to/training_checkpoint
OUT_DIR=/path/to/converted_model

python scripts/convert_gr_checkpoint.py \
    --input-path "$CKPT_DIR" \
    --output-dir "$OUT_DIR" \
    --dataset movielens-1m
```

NPU 侧 examples 的现场执行清单见 [EXAMPLES_NPU_RUNBOOK.md](/root/PROJECT/HSTU_Update/GR_Repo_main/EXAMPLES_NPU_RUNBOOK.md)。

### Uninstallation

```bash
# Uninstall vllm-GR
pip uninstall vllm-gr
```

如果是平台专用环境，通常还需要按环境分别卸载 GPU/NPU 侧附加依赖。

### Basic Usage

```python
from vllm_gr import GRLLM, GRPrompt

# Initialize GRLLM
llm = GRLLM(
    model="path/to/hstu_model",  # Model path
    max_seq_len=512,             # Max user history length
    max_candidates=100,          # Max candidates per request
    default_top_k=10,            # Default recommendations count
)

# Prepare request using GRPrompt format
prompt: GRPrompt = {
    "user_id": "user_001",
    "user_history": [
        {"item_id": 1001, "timestamp": 1640995200},
        {"item_id": 1002, "timestamp": 1640995300},
        {"item_id": 1003, "timestamp": 1640995400},
    ],
    "candidates": [
        {"item_id": 2001},
        {"item_id": 2002},
        {"item_id": 2003},
    ],
}

# Generate recommendations
outputs = llm.recommend(
    prompts=[prompt],
    use_tqdm=False,
    top_k=10,
)

# Access results - outputs is List[GRRequestOutput]
output = outputs[0]
print(f"Request ID: {output.request_id}")
print(f"User ID: {output.user_id}")

for item in output.recommendations:
    print(f"#{item.rank}: item_{item.item_id} (score: {item.score:.4f})")
```

### Batch Recommendations

```python
from vllm_gr import GRLLM, GRPrompt
from typing import List

llm = GRLLM(model="path/to/model")

# Create multiple prompts
prompts: List[GRPrompt] = [
    {
        "user_id": f"user_{i}",
        "user_history": [{"item_id": j} for j in range(10)],
        "candidates": [{"item_id": k * 10} for k in range(20)],
    }
    for i in range(100)
]

# Method 1: Process all at once
outputs = llm.recommend(prompts, top_k=5)

# Method 2: Process in batches (for large workloads)
outputs = llm.recommend_batch(prompts, batch_size=32, top_k=5)

# Access results
for output in outputs:
    print(f"{output.user_id}: {output.item_ids[:3]}...")
```

### Using Config-Based Model (No Pretrained Weights)

You can create a randomly initialized HSTU model from configuration for testing or development:

```python
from vllm_gr import GRLLM, GRPrompt

# Create GRLLM with model architecture config (no pretrained weights needed)
llm = GRLLM(
    model="",                       # Empty - use config to create model
    max_seq_len=256,
    max_candidates=50,
    default_top_k=10,
    # Model architecture parameters (used when model="" or model_path is empty)
    embedding_dim=128,              # Embedding dimension
    hash_size=100_000,              # Embedding table size
    num_layers=2,                   # HSTU layers (smaller for testing)
    num_heads=2,                    # Attention heads
    hidden_dim=64,                  # Hidden dimension
    attention_dim=32,               # Attention dimension
)

# Use as normal - model will be randomly initialized
prompt: GRPrompt = {
    "user_id": "test_user",
    "user_history": [{"item_id": i} for i in range(10)],
    "candidates": [{"item_id": i * 10} for i in range(20)],
}

outputs = llm.recommend([prompt], top_k=5)
print(f"Got {len(outputs[0].recommendations)} recommendations")
```

This is useful for:
- **Testing**: Verify inference pipeline without real model weights
- **Development**: Debug and profile model architecture
- **Benchmarking**: Measure inference performance

## Core Types

### GRPrompt (Input)

```python
from vllm_gr import GRPrompt

prompt: GRPrompt = {
    "user_id": "user_123",            # Required: User identifier
    "user_history": [                  # Required: Interaction history
        {"item_id": 100, "timestamp": 1000, "rating": 5},
        {"item_id": 200, "timestamp": 2000},
    ],
    "candidates": [                    # Required: Items to rank
        {"item_id": 300},
        {"item_id": 400},
    ],
    "num_recommendations": 10,         # Optional: Override top_k
    "context": {"device": "mobile"},   # Optional: Additional context
}
```

### GRRequestOutput (Output)

```python
from vllm_gr import GRRequestOutput, RecommendedItem

# Output structure
output: GRRequestOutput
output.request_id      # str: Request identifier
output.user_id         # str|int: User identifier
output.recommendations # List[RecommendedItem]: Ranked items
output.finished        # bool: Completion status
output.all_scores      # Optional[Tensor]: All candidate scores
output.metrics         # Dict: Performance metrics

# RecommendedItem structure
item: RecommendedItem
item.item_id           # str|int: Item identifier
item.score             # float: Prediction score
item.rank              # int: Rank (1-indexed)
item.features          # Optional[Dict]: Item features

# Serialization
output_dict = output.to_dict()  # Convert to JSON-serializable dict
```

## Architecture

### Inheritance Design

vllm-GR classes inherit from vLLM base classes:

- `GRLLM` inherits from `vllm.entrypoints.llm.LLM`
- `GRLLMEngine` inherits from `vllm.v1.engine.llm_engine.LLMEngine`
- `GRInputProcessor` inherits from `vllm.v1.engine.input_processor.InputProcessor`
- `GROutputProcessor` inherits from `vllm.v1.engine.output_processor.OutputProcessor`
- `GRRequestState` inherits from `vllm.v1.engine.output_processor.RequestState`
- `GRRequestOutput` inherits from `vllm.outputs.PoolingRequestOutput`
- `GREngineArgs` inherits from `vllm.engine.arg_utils.EngineArgs`

### Key Differences from vLLM LLM

| Aspect | vLLM LLM | GRLLM |
|--------|----------|-------|
| Input | TokensPrompt/TextPrompt | GRPrompt (user history + candidates) |
| Output | RequestOutput (token sequence) | GRRequestOutput (ranked items) |
| step() | Generates one token | Evaluates all candidates at once |
| Model | Language Model (GPT, LLaMA) | HSTU (generative recommender) |

## API Reference

### GRLLM

```python
class GRLLM(LLM):
    def __init__(
        self,
        model: str = "",                    # Model path (optional)
        embedding_path: Optional[str] = None,
        max_seq_len: int = 512,             # Max user history length
        max_candidates: int = 1000,         # Max candidates
        default_top_k: int = 10,            # Default recommendations
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        # Model loading options
        device: Optional[str] = None,       # "cuda" or "cpu"
        embedding_dim: int = 192,           # Embedding dimension
        hash_size: int = 10_000_000,        # Embedding table size
        load_model: bool = True,            # Enable model loading
        # HSTU architecture (for config-based model creation when no model_path)
        num_layers: int = 3,                # HSTU layers
        num_heads: int = 2,                 # Attention heads
        hidden_dim: int = 128,              # Hidden dimension
        attention_dim: int = 64,            # Attention dimension
        **kwargs,
    ):
        """Initialize GRLLM."""
    
    def recommend(
        self,
        prompts: Union[GRPrompt, Sequence[GRPrompt]],
        *,
        use_tqdm: bool = True,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """
        Generate recommendations.
        
        Args:
            prompts: Single or batch of GRPrompt
            use_tqdm: Show progress bar
            top_k: Override default recommendations count
        
        Returns:
            List of GRRequestOutput, one per input prompt
        """
    
    def recommend_batch(
        self,
        prompts: Sequence[GRPrompt],
        batch_size: int = 32,
        top_k: Optional[int] = None,
    ) -> List[GRRequestOutput]:
        """
        Batch recommendations for large workloads.
        
        Processes prompts in chunks of batch_size.
        """
```

### GRLLMEngine

```python
from vllm_gr import GRLLMEngine, GREngineArgs
from vllm.pooling_params import PoolingParams

# Create engine directly (advanced usage)
args = GREngineArgs(
    model="path/to/model",
    max_seq_len=256,
    max_candidates=100,
)
engine = GRLLMEngine.from_engine_args(args)

# Add request (uses standard LLMEngine.add_request with PoolingParams)
engine.add_request(
    request_id="req-001",
    prompt=prompt,
    params=PoolingParams(task="embed"),
)

# Run inference
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        print(output)
```

## Project Structure

```
vllm_gr/
├── __init__.py                 # Package exports
├── inputs/
│   └── data.py                 # GRPrompt, GRInputs types
├── outputs.py                  # GRRequestOutput, RecommendedItem
├── request.py                  # GRRequest class
├── patch.py                    # Monkey patching utilities
├── entrypoint/
│   └── gr.py                   # GRLLM main interface (inherits vLLM LLM)
├── engine/
│   ├── arg_utils.py            # GREngineArgs (inherits vLLM EngineArgs)
│   ├── gr_engine.py            # GRLLMEngine (inherits vLLM LLMEngine)
│   ├── gr_processor.py         # Input processor
│   ├── gr_output_processor.py  # Output processor
│   └── model_runner.py         # Model runner for HSTU inference
├── config/                     # Model configuration
├── model_executor/             # Model execution
│   ├── models/                 # HSTUForRecommendation (unified model)
│   │   ├── hstu_vllm.py        # vLLM-compatible HSTU model
│   │   └── registry.py         # GRModelRegistry
│   └── model_loader.py         # Model loading
└── sample/                     # Sampling utilities
```

## Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_engine.py -v          # Engine, config, data types
python -m pytest tests/test_processors.py -v      # Input/output processors
python -m pytest tests/test_weight_loading.py -v  # Checkpoint conversion & weight loading

# Unified runner (unit tests + optional examples)
python scripts/run_tests.py                       # Unit tests only
python scripts/run_tests.py --all                 # Unit tests + examples
```

## Examples

See the `examples/` directory:

```bash
# Simple inference
python examples/simple_inference.py

# Batch inference
python examples/batch_inference.py
```

## Development Roadmap

### Phase 1: Core Types (Completed)
- GRPrompt, GRInputs, GRRequestOutput
- GRRequest with Monkey Patch
- Unit tests

### Phase 2: Processors (Completed)
- GRProcessor (input encoding)
- GROutputProcessor (Top-K sampling)
- Processor tests

### Phase 3: Engine & Interface (Completed)
- GRLLMEngine inheriting vLLM LLMEngine
- GRLLM inheriting vLLM LLM
- End-to-end tests

### Phase 4: Model Integration (In Progress)
- HSTU model loading
- GPU inference
- Config-based model creation

### Phase 5: Online Service (Planned)
- API server
- Continuous batching
- Monitoring

## License

Apache License 2.0

The DLRM_v3 model code is ported from Meta's [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) repository.

## Acknowledgements

- [Meta's generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
- [vLLM](https://github.com/vllm-project/vllm)
