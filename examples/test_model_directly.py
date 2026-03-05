#!/usr/bin/env python3
"""
Test model creation directly (bypass all vLLM and GRLLM code).

Verifies the HSTU model can be created and performs a dummy forward pass.
After the refactoring, DlrmHSTU owns EmbeddingCollection internally
and forward() accepts KeyedJaggedTensor inputs.
"""

import sys
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

print("=" * 60)
print("Testing HSTU model creation directly")
print("=" * 60)

# Step 1: Basic imports
print("\n[Step 1] Basic imports...")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("  WARNING: CUDA not available, testing on CPU only")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    print(f"  Using device: {device}")

    try:
        torch.cuda.set_device(0)
        x = torch.ones(10, device=device)
        print("  Basic CUDA test: PASSED")
    except Exception as e:
        print(f"  Basic CUDA test: FAILED - {e}")
        sys.exit(1)

# Step 2: Import model components
print("\n[Step 2] Importing model components...")
try:
    from vllm_gr.config.hstu_hf_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
    print("  Imports successful")
except Exception as e:
    print(f"  Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create model config
print("\n[Step 3] Creating model config...")
config = HSTUConfig(
    num_layers=2,
    num_heads=2,
    embedding_dim=64,
    attention_dim=32,
    hidden_dim=64,
    embedding_table_dim=64,
    max_seq_len=64,
    max_num_candidates=50,
    hash_size=10000,
    user_embedding_feature_names=["item_id"],
    item_embedding_feature_names=["item_candidate_id"],
    uih_post_id_feature_name="item_id",
    embedding_tables=[
        {
            "name": "item_embedding",
            "embedding_dim": 64,
            "num_embeddings": 10000,
            "feature_names": ["item_id", "item_candidate_id"],
        },
    ],
)
print(f"  Config created: hash_size={config.hash_size}, dim={config.embedding_dim}")

# Step 4: Create model on CPU
print("\n[Step 4] Creating model on CPU...")
try:
    model = HSTUForRecommendation(config=config)
    print("  Model created on CPU successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
except Exception as e:
    print(f"  Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Move to CUDA (if available)
if torch.cuda.is_available():
    print("\n[Step 5] Moving model to CUDA...")
    try:
        torch.cuda.synchronize()
        model = model.to(device)
        print("  Model moved to CUDA")
        torch.cuda.synchronize()
        print("  CUDA sync after move: OK")
    except Exception as e:
        print(f"  Failed to move model to CUDA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Step 6: Test dummy forward pass (no KJT features → returns zeros)
print("\n[Step 6] Testing dummy forward pass (no KJT features)...")
try:
    model.eval()
    input_ids = torch.tensor([1, 2, 3], device=device)
    positions = torch.tensor([0, 1, 2], device=device)

    with torch.no_grad():
        output = model.forward(input_ids, positions)

    print(f"  Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (3, 64), f"Expected (3, 64), got {output.shape}"
    print("  Shape assertion: PASSED")
except Exception as e:
    print(f"  Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests PASSED! Model works correctly.")
print("=" * 60)
