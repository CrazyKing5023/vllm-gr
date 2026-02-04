# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
End-to-End Offline Inference Tests

Tests the complete vllm-GR inference flow from JSON request to recommendation response.
"""

import os
import sys
import time

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_creation():
    """Test HSTUModelConfig creation and serialization"""
    print("\n=== Test: Config Creation ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig
    
    # Test default config
    config = HSTUModelConfig()
    assert config.num_layers == 5
    assert config.num_heads == 4
    assert config.embedding_dim == 512
    assert len(config.task_names) == 2
    print(f"  ✓ Default config created: num_layers={config.num_layers}, num_heads={config.num_heads}")
    
    # Test custom config
    custom_config = HSTUModelConfig(
        num_layers=3,
        num_heads=2,
        embedding_dim=256,
        max_seq_len=512,
        task_names=["click", "like", "share"],
    )
    assert custom_config.num_layers == 3
    assert len(custom_config.task_names) == 3
    print(f"  ✓ Custom config created: tasks={custom_config.task_names}")
    
    # Test config serialization
    config_dict = config.to_dict()
    assert "num_layers" in config_dict
    assert "embedding_dim" in config_dict
    print(f"  ✓ Config serialization successful")
    
    print("✅ Config creation test passed")


def test_embedding_manager():
    """Test EmbeddingManager creation and lookup"""
    print("\n=== Test: Embedding Manager ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.model_executor.embedding_manager import EmbeddingManager
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HSTUModelConfig(
        hash_size=10000,
        embedding_table_dim=64,
    )
    
    embedding_manager = EmbeddingManager(config, device)
    print(f"  ✓ EmbeddingManager created, device: {device}")
    
    # Test embedding lookup
    indices = torch.tensor([1, 5, 10, 100], dtype=torch.long, device=device)
    embeddings = embedding_manager.lookup("item_id", indices)
    
    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] == config.embedding_table_dim
    print(f"  ✓ Embedding lookup successful: shape={embeddings.shape}")
    
    print("✅ Embedding manager test passed")


def test_input_processor():
    """Test GRInputProcessor"""
    print("\n=== Test: Input Processor ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.engine.input_processor import GRInputProcessor
    
    config = HSTUModelConfig(hash_size=10000)
    processor = GRInputProcessor(config)
    
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
        {"item_id": "movie_103"},
        {"item_id": "movie_104"},
    ]
    
    seq_indices, seq_lengths, cand_indices, num_candidates = processor.process(
        user_id=user_id,
        sequence=sequence,
        candidates=candidates,
    )
    
    assert seq_indices.shape[0] == 3
    assert seq_lengths.item() == 3
    assert cand_indices.shape[0] == 5
    assert num_candidates.item() == 5
    print(f"  ✓ Input processing successful: seq_len={seq_lengths.item()}, num_candidates={num_candidates.item()}")
    
    print("✅ Input processor test passed")


def test_output_processor():
    """Test GROutputProcessor"""
    print("\n=== Test: Output Processor ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.engine.output_processor import GROutputProcessor
    
    config = HSTUModelConfig()
    processor = GROutputProcessor(config)
    
    sampled = [
        ("movie_102", 0.95, [0.9, 0.8]),
        ("movie_100", 0.88, [0.85, 0.75]),
        ("movie_101", 0.76, [0.7, 0.65]),
    ]
    task_names = ["is_click", "is_like"]
    
    recommendations = processor.format_output(sampled, task_names)
    
    assert len(recommendations) == 3
    assert recommendations[0]["item_id"] == "movie_102"
    assert recommendations[0]["rank"] == 0
    assert "task_scores" in recommendations[0]
    print(f"  ✓ Output formatting successful: {len(recommendations)} recommendations")
    
    print("✅ Output processor test passed")


def test_topk_sampler():
    """Test TopKSampler"""
    print("\n=== Test: Top-K Sampler ===")
    
    from vllm_gr.sample.topk_sampler import TopKSampler
    
    sampler = TopKSampler()
    
    # Create mock predictions (2 tasks, 5 candidates)
    predictions = torch.tensor([
        [0.9, 0.5, 0.7, 0.3, 0.8],
        [0.85, 0.6, 0.65, 0.4, 0.75],
    ])
    candidate_ids = ["item_0", "item_1", "item_2", "item_3", "item_4"]
    
    results = sampler.sample(predictions, candidate_ids, top_k=3)
    
    assert len(results) == 3
    assert all(len(r) == 3 for r in results)  # (id, score, task_scores)
    print(f"  ✓ Top-K sampling successful: top_3={[r[0] for r in results]}")
    
    print("✅ Top-K sampler test passed")


def test_hstu_model_wrapper():
    """Test HSTUModelWrapper creation and forward pass"""
    print("\n=== Test: HSTU Model Wrapper ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.model_executor.embedding_manager import EmbeddingManager
    from vllm_gr.model_executor.models.hstu import HSTUModelWrapper
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use small config for testing
    config = HSTUModelConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=128,
        attention_dim=32,
        hidden_dim=64,
        embedding_table_dim=64,
        preprocessor_hidden_dim=64,
        max_seq_len=64,
        max_num_candidates=10,
        hash_size=1000,
    )
    
    embedding_manager = EmbeddingManager(config, device)
    model = HSTUModelWrapper(config, embedding_manager, device)
    model = model.to(device)
    model.eval()
    print(f"  ✓ HSTUModelWrapper created, device: {device}")
    
    # Create mock input
    seq_len = 5
    num_candidates = 3
    
    uih_embeddings = torch.randn(seq_len, config.embedding_table_dim, device=device)
    uih_lengths = torch.tensor([seq_len], dtype=torch.long, device=device)
    candidate_embeddings = torch.randn(num_candidates, config.embedding_table_dim, device=device)
    num_candidates_tensor = torch.tensor([num_candidates], dtype=torch.long, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(
            uih_embeddings=uih_embeddings,
            uih_lengths=uih_lengths,
            candidate_embeddings=candidate_embeddings,
            num_candidates=num_candidates_tensor,
        )
    
    assert output.predictions is not None
    assert output.user_embeddings is not None
    assert output.item_embeddings is not None
    print(f"  ✓ Forward pass successful: predictions shape={output.predictions.shape}")
    
    print("✅ HSTU model wrapper test passed")


def test_grllm_basic():
    """Test GRLLM basic initialization and recommendation functionality"""
    print("\n=== Test: GRLLM Basic Functionality ===")
    
    from vllm_gr.entrypoint.gr import GRLLM
    from vllm_gr.config.hstu_config import HSTUModelConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use minimal config for fast testing
    config = HSTUModelConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=128,
        attention_dim=32,
        hidden_dim=64,
        embedding_table_dim=64,
        preprocessor_hidden_dim=64,
        max_seq_len=64,
        max_num_candidates=20,
        hash_size=1000,
    )
    
    # Create GRLLM (no pretrained weights)
    llm = GRLLM(model_path=None, config=config, device=device)
    print(f"  ✓ GRLLM created, device: {device}")
    
    # Prepare request
    user_id = "user_001"
    sequence = [
        {"item_id": "movie_123", "timestamp": 1640995200},
        {"item_id": "movie_456", "timestamp": 1640995300},
        {"item_id": "movie_789", "timestamp": 1640995400},
    ]
    candidates = [{"item_id": f"candidate_{i}"} for i in range(10)]
    
    # Generate recommendations
    start_time = time.time()
    response = llm.recommend(
        user_id=user_id,
        sequence=sequence,
        candidates=candidates,
        top_k=5,
    )
    elapsed = (time.time() - start_time) * 1000
    
    assert response.request_id is not None
    assert len(response.recommendations) == 5
    assert response.latency_ms > 0
    print(f"  ✓ Recommendations generated, latency: {elapsed:.2f}ms (reported: {response.latency_ms:.2f}ms)")
    print(f"  ✓ Top 5 recommendations: {[r['item_id'] for r in response.recommendations[:3]]}...")
    
    for i, rec in enumerate(response.recommendations):
        assert rec["rank"] == i
        assert "item_id" in rec
        assert "score" in rec
    
    print("✅ GRLLM basic functionality test passed")


def test_grllm_batch():
    """Test GRLLM batch recommendations"""
    print("\n=== Test: GRLLM Batch Recommendations ===")
    
    from vllm_gr.entrypoint.gr import GRLLM
    from vllm_gr.config.hstu_config import HSTUModelConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = HSTUModelConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=128,
        attention_dim=32,
        hidden_dim=64,
        embedding_table_dim=64,
        preprocessor_hidden_dim=64,
        max_seq_len=64,
        max_num_candidates=20,
        hash_size=1000,
    )
    
    llm = GRLLM(model_path=None, config=config, device=device)
    
    # Prepare batch requests
    requests = [
        {
            "user_id": f"user_{i}",
            "sequence": [{"item_id": f"item_{j}", "timestamp": 1000 + j} for j in range(5)],
            "candidates": [{"item_id": f"cand_{k}"} for k in range(8)],
        }
        for i in range(3)
    ]
    
    # Batch recommendations
    start_time = time.time()
    responses = llm.recommend_batch(requests, top_k=5)
    elapsed = (time.time() - start_time) * 1000
    
    assert len(responses) == 3
    for response in responses:
        assert len(response.recommendations) == 5
    
    print(f"  ✓ Batch recommendations completed: 3 users, latency {elapsed:.2f}ms")
    print("✅ GRLLM batch recommendations test passed")


def test_full_e2e_pipeline():
    """Full end-to-end test simulating real use case"""
    print("\n=== Test: Full End-to-End Pipeline ===")
    
    from vllm_gr import GRLLM
    from vllm_gr.config.hstu_config import HSTUModelConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Create configuration
    config = HSTUModelConfig(
        num_layers=2,
        num_heads=2,
        embedding_dim=128,
        attention_dim=32,
        hidden_dim=64,
        embedding_table_dim=64,
        max_seq_len=128,
        max_num_candidates=50,
        hash_size=10000,
        task_names=["click", "like"],
    )
    print("  ✓ Step 1: Configuration created")
    
    # 2. Initialize GRLLM
    llm = GRLLM(config=config, device=device)
    print(f"  ✓ Step 2: GRLLM initialized, device: {device}")
    
    # 3. Simulate real request
    user_id = "user_alice_12345"
    sequence = [
        {"item_id": "The_Matrix_1999", "timestamp": 1609459200},
        {"item_id": "Inception_2010", "timestamp": 1609545600},
        {"item_id": "Interstellar_2014", "timestamp": 1609632000},
        {"item_id": "Blade_Runner_2049", "timestamp": 1609718400},
        {"item_id": "Dune_2021", "timestamp": 1609804800},
    ]
    candidates = [
        {"item_id": "Arrival_2016"},
        {"item_id": "Ex_Machina_2014"},
        {"item_id": "Annihilation_2018"},
        {"item_id": "Tenet_2020"},
        {"item_id": "Avatar_2009"},
        {"item_id": "Edge_of_Tomorrow_2014"},
        {"item_id": "District_9_2009"},
        {"item_id": "Elysium_2013"},
        {"item_id": "Gravity_2013"},
        {"item_id": "The_Martian_2015"},
    ]
    
    # 4. Get recommendations
    response = llm.recommend(
        user_id=user_id,
        sequence=sequence,
        candidates=candidates,
        top_k=5,
        request_id="test_request_001",
    )
    print(f"  ✓ Step 3: Recommendations generated")
    
    # 5. Validate response
    assert response.request_id == "test_request_001"
    assert len(response.recommendations) == 5
    assert all("item_id" in r and "score" in r and "rank" in r for r in response.recommendations)
    
    print(f"\n  📊 Results:")
    print(f"     Request ID: {response.request_id}")
    print(f"     Latency: {response.latency_ms:.2f}ms")
    print(f"     Top 5 Recommendations:")
    for rec in response.recommendations:
        print(f"       #{rec['rank']}: {rec['item_id']} (score: {rec['score']:.4f})")
    
    print("\n✅ Full end-to-end pipeline test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print(" vllm-GR Offline Inference End-to-End Test Suite")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    
    tests = [
        ("Config Creation", test_config_creation),
        ("Embedding Manager", test_embedding_manager),
        ("Input Processor", test_input_processor),
        ("Output Processor", test_output_processor),
        ("Top-K Sampler", test_topk_sampler),
        ("HSTU Model Wrapper", test_hstu_model_wrapper),
        ("GRLLM Basic Functionality", test_grllm_basic),
        ("GRLLM Batch Recommendations", test_grllm_batch),
        ("Full End-to-End Pipeline", test_full_e2e_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  Some tests failed, please check the error messages above.")
        return False
    else:
        print("\n🎉 All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
