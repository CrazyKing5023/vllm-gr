# Demo HSTU Model for vllm-GR

This directory contains a minimal HSTU model configuration for testing and demonstration purposes.

## Purpose

The `demo-model` directory serves as a **demo/testing model path** that:

1. Contains model architecture configuration (`config.json`)
2. Does **NOT** contain actual model weights
3. Allows vllm-GR to run full inference pipeline with randomly initialized weights

This is useful for:
- Testing the complete inference pipeline without real model weights
- Development and debugging
- CI/CD testing
- Demonstration of the vllm-GR API

## Directory Structure

```
demo-model/
├── config.json    # Model architecture configuration
├── README.md      # This file
└── (no weights)   # Weights files are optional
```

## Model Loading Behavior

When vllm-GR loads a model from this directory:

1. **Config Loading**: Reads `config.json` to get model architecture parameters
2. **Weights Loading**: Looks for weight files in this order:
   - `pytorch_model.bin`
   - `model.safetensors`
   - `model.pt`
   - `weights.pt`
3. **Fallback**: If no weights file exists, uses randomly initialized weights

## Usage

```python
import vllm_gr  # Auto-registers HSTUForRecommendation
from vllm_gr import GRLLM

# Option 1: Use demo-model directory (loads config, initializes random weights)
llm = GRLLM(model="examples/demo-model")

# Option 2: Override config parameters for smaller/faster demo
llm = GRLLM(
    model="examples/demo-model",
    embedding_dim=64,    # Override from config
    hash_size=10000,
    num_layers=2,
    num_heads=2,
)

# Option 3: No model path (uses only provided parameters)
llm = GRLLM(
    model="",
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
  "attention_dim": 64,
  "hidden_dim": 128,
  "hash_size": 10000000,
  "max_seq_len": 256,
  "max_num_candidates": 50
}
```

## Adding Real Weights (Production)

To use this directory structure with real trained weights:

1. Train your HSTU model
2. Save weights to one of the supported formats:
   ```python
   # PyTorch format
   torch.save(model.state_dict(), "demo-model/pytorch_model.bin")
   
   # Or using safetensors
   from safetensors.torch import save_file
   save_file(model.state_dict(), "demo-model/model.safetensors")
   ```
3. vllm-GR will automatically load the weights on startup

## Notes

- This demo uses random initialization - **scores are not meaningful**
- The model runs actual DlrmHSTU forward pass (not mocked)
- Useful for validating the complete inference pipeline
- For production, add trained weights alongside config.json
