#!/usr/bin/env python3
"""
Test GR-native mode directly (bypass vLLM EngineCore).

This script helps diagnose whether the CUDA error is in:
1. vLLM's EngineCore initialization, or
2. Our model creation code

Run this to see if GR-native mode works when we skip EngineCore entirely.
"""

import sys
import os

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

print("=" * 60)
print("Testing GR-native mode (bypassing vLLM EngineCore)")
print("=" * 60)

# Test 1: Check CUDA availability
print("\n[Step 1] Checking CUDA availability...")
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    # Test basic CUDA operation
    try:
        x = torch.ones(10, device='cuda')
        y = x + 1
        print(f"  Basic CUDA test: PASSED")
    except Exception as e:
        print(f"  Basic CUDA test: FAILED - {e}")
        sys.exit(1)

# Test 2: Import vllm_gr
print("\n[Step 2] Importing vllm_gr...")
try:
    from vllm_gr import GRLLM, GRPrompt
    print("  Import successful")
except Exception as e:
    print(f"  Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create GRLLM with force_gr_native=True
print("\n[Step 3] Creating GRLLM with force_gr_native=True...")
DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "demo-model")
print(f"  Model path: {DEMO_MODEL_PATH}")

try:
    llm = GRLLM(
        model=DEMO_MODEL_PATH,
        max_seq_len=64,
        max_candidates=50,
        default_top_k=10,
        embedding_dim=64,
        hash_size=10000,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        attention_dim=32,
        max_batch_size=16,
        scheduling_policy="fcfs",
        force_gr_native=True,  # Skip EngineCore
    )
    print("  GRLLM created successfully!")
    print(f"  Execution mode: GR-native (forced)")
except Exception as e:
    print(f"  GRLLM creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run a simple recommendation
print("\n[Step 4] Running recommendation...")
prompt: GRPrompt = {
    "user_id": "test_user",
    "user_history": [
        {"item_id": 1, "timestamp": 1000},
        {"item_id": 2, "timestamp": 2000},
    ],
    "candidates": [{"item_id": i} for i in range(1, 20)],
    "num_recommendations": 5,
}

try:
    outputs = llm.recommend([prompt])
    print("  Recommendation completed!")
    print(f"  Number of results: {len(outputs)}")
    if outputs:
        print(f"  Top recommendations:")
        for item in outputs[0].recommendations[:5]:
            print(f"    Rank {item.rank}: item_{item.item_id} (score: {item.score:.4f})")
except Exception as e:
    print(f"  Recommendation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests PASSED! GR-native mode is working.")
print("=" * 60)
print("\nThe CUDA error is likely in vLLM's EngineCore initialization,")
print("not in our model code. We may need to investigate:")
print("1. vLLM version compatibility")
print("2. HSTUConfig registration with AutoConfig")
print("3. vLLM's model loading process")
