# Demo HSTU Model for vllm-GR

This directory contains a minimal HSTU model configuration for testing.

## vLLM Model Registration

vllm-GR follows the vllm-omni pattern for model registration:

1. **architectures**: `["HSTUForRecommendation"]` - Registered with vLLM ModelRegistry
2. **model_type**: `"hstu"` - For config.json compatibility

Models are automatically registered when you import `vllm_gr`.

## Usage

```python
import vllm_gr  # Auto-registers HSTUForRecommendation
from vllm_gr import GRLLM

# Option 1: Use demo-model directory (loads config, initializes random weights)
llm = GRLLM(model="examples/demo-model")

# Option 2: No model path (uses default config with random weights)
llm = GRLLM(
    model=None,
    embedding_dim=64,
    hash_size=10000,
    num_layers=2,
    num_heads=2,
)
```

## config.json Format

```json
{
  "model_type": "hstu",
  "architectures": ["HSTUForRecommendation"],
  "num_layers": 3,
  "num_heads": 2,
  "embedding_dim": 192,
  ...
}
```

## Notes

- This demo has no pretrained weights (uses random initialization)
- The model runs actual DlrmHSTU forward pass
- For production, add trained weights (.pt, .pth, .bin, or .safetensors) alongside config.json
