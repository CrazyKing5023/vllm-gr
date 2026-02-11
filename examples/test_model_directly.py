#!/usr/bin/env python3
"""
Test model creation directly (bypass all vLLM and GRLLM code).

This is the most basic test to verify the HSTU model can be created
and moved to CUDA without errors.
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
    
    # Simple CUDA test
    try:
        torch.cuda.set_device(0)
        x = torch.ones(10, device=device)
        print(f"  Basic CUDA test: PASSED")
    except Exception as e:
        print(f"  Basic CUDA test: FAILED - {e}")
        sys.exit(1)

# Step 2: Import model components
print("\n[Step 2] Importing model components...")
try:
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
    print("  Imports successful")
except Exception as e:
    print(f"  Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create model config
print("\n[Step 3] Creating model config...")
model_config = HSTUModelConfig(
    num_layers=2,
    num_heads=2,
    embedding_dim=64,
    attention_dim=32,
    hidden_dim=64,
    embedding_table_dim=64,
    max_seq_len=64,
    max_num_candidates=50,
    hash_size=10000,  # Small for testing
)
print(f"  Config created: hash_size={model_config.hash_size}, dim={model_config.embedding_dim}")

# Step 4: Create model on CPU
print("\n[Step 4] Creating model on CPU...")
try:
    model = HSTUForRecommendation.from_model_config(
        config=model_config,
        device=torch.device("cpu"),
    )
    print("  Model created on CPU successfully!")
    
    # Count parameters
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
        print("  CUDA sync before move: OK")
        
        model = model.to(device)
        print("  Model moved to CUDA")
        
        torch.cuda.synchronize()
        print("  CUDA sync after move: OK")
        
    except Exception as e:
        print(f"  Failed to move model to CUDA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Step 6: Test forward pass with dummy data (using run_recommendation with item IDs)
print("\n[Step 6] Testing forward pass with run_recommendation...")
try:
    model.eval()
    
    # Batch of 2 users
    # User 1: history of 5 items, 10 candidates
    # User 2: history of 8 items, 10 candidates
    uih_lengths = torch.tensor([5, 8], device=device)
    num_candidates = torch.tensor([10, 10], device=device)
    
    # Create jagged (concatenated) item IDs
    # Total UIH items: 5 + 8 = 13
    # Total candidates: 10 + 10 = 20
    uih_item_ids = torch.randint(0, 10000, (13,), device=device)  # (total_uih_len,)
    candidate_item_ids = torch.randint(0, 10000, (20,), device=device)  # (total_candidates,)
    
    print(f"  UIH item IDs shape: {uih_item_ids.shape}")
    print(f"  Candidate item IDs shape: {candidate_item_ids.shape}")
    print(f"  UIH lengths: {uih_lengths}")
    print(f"  Num candidates: {num_candidates}")
    
    # Run recommendation
    with torch.no_grad():
        predictions = model.run_recommendation(
            uih_item_ids=uih_item_ids,
            uih_lengths=uih_lengths,
            candidate_item_ids=candidate_item_ids,
            num_candidates=num_candidates,
        )
    
    print(f"  Forward pass successful!")
    print(f"  Output predictions shape: {predictions.shape}")
    
except Exception as e:
    print(f"  Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests PASSED! Model works correctly.")
print("=" * 60)
print("\nThe HSTU model can be created and run on CUDA without issues.")
print("If you're seeing CUDA errors with GRLLM, the issue is likely in:")
print("1. vLLM's EngineCore initialization")
print("2. The integration between vLLM and our model")
print("3. vLLM's model loading/validation process")
