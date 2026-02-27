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

## Dependencies

### Required Dependencies
- **PyTorch** (`torch>=2.0.0`): Required for all model inference
- **tqdm** (`tqdm>=4.60.0`): Required for progress display
- **vLLM** (`vllm>=0.14.1`): Required for LLM/LLMEngine base classes

### Optional Dependencies
- **TorchRec** (`torchrec>=0.6.0`): Enables KeyedJaggedTensor support for optimized embedding lookup
- **SafeTensors** (`safetensors>=0.4.0`): For model weight loading

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-GR.git
cd vllm-GR

# Standard install (includes torch, tqdm, vllm)
pip install .

# Development install (editable mode)
pip install -e ".[dev]"

# Full install with all dependencies
pip install ".[all]"
```

### Uninstallation

```bash
# Uninstall vllm-GR
pip uninstall vllm-gr
```

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
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_data_types.py -v      # Core data types
python -m pytest tests/test_processors.py -v      # Input/output processors
python -m pytest tests/test_engine.py -v          # Engine components
python -m pytest tests/test_model_runner.py -v    # Model runner
python -m pytest tests/test_offline_inference.py -v  # End-to-end tests
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
