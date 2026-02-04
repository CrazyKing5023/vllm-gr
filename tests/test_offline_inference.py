# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
离线推理端到端测试

测试 vllm-GR 完整的推理流程，从 JSON 请求到推荐结果响应。
"""

import os
import sys
import time

import torch

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_creation():
    """测试 HSTUModelConfig 创建和序列化"""
    print("\n=== 测试: 配置创建 ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig, EmbeddingTableConfig
    
    # 测试默认配置
    config = HSTUModelConfig()
    assert config.num_layers == 5
    assert config.num_heads == 4
    assert config.embedding_dim == 512
    assert len(config.task_names) == 2
    print(f"  ✓ 默认配置创建成功: num_layers={config.num_layers}, num_heads={config.num_heads}")
    
    # 测试自定义配置
    custom_config = HSTUModelConfig(
        num_layers=3,
        num_heads=2,
        embedding_dim=256,
        max_seq_len=512,
        task_names=["click", "like", "share"],
    )
    assert custom_config.num_layers == 3
    assert len(custom_config.task_names) == 3
    print(f"  ✓ 自定义配置创建成功: tasks={custom_config.task_names}")
    
    # 测试配置序列化
    config_dict = config.to_dict()
    assert "num_layers" in config_dict
    assert "embedding_dim" in config_dict
    print(f"  ✓ 配置序列化正常")
    
    print("✅ 配置创建测试通过")


def test_embedding_manager():
    """测试 EmbeddingManager 创建和查找"""
    print("\n=== 测试: 嵌入管理器 ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.model_executor.embedding_manager import EmbeddingManager
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HSTUModelConfig(
        hash_size=10000,
        embedding_table_dim=64,
    )
    
    embedding_manager = EmbeddingManager(config, device)
    print(f"  ✓ EmbeddingManager 创建成功，设备: {device}")
    
    # 测试嵌入查找
    indices = torch.tensor([1, 5, 10, 100], dtype=torch.long, device=device)
    embeddings = embedding_manager.lookup("item_id", indices)
    
    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] == config.embedding_table_dim
    print(f"  ✓ 嵌入查找正常: shape={embeddings.shape}")
    
    print("✅ 嵌入管理器测试通过")


def test_input_processor():
    """测试 GRInputProcessor"""
    print("\n=== 测试: 输入处理器 ===")
    
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
    print(f"  ✓ 输入处理正常: seq_len={seq_lengths.item()}, num_candidates={num_candidates.item()}")
    
    print("✅ 输入处理器测试通过")


def test_output_processor():
    """测试 GROutputProcessor"""
    print("\n=== 测试: 输出处理器 ===")
    
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
    print(f"  ✓ 输出格式化正常: {len(recommendations)} 条推荐")
    
    print("✅ 输出处理器测试通过")


def test_topk_sampler():
    """测试 TopKSampler"""
    print("\n=== 测试: Top-K 采样器 ===")
    
    from vllm_gr.sample.topk_sampler import TopKSampler
    
    sampler = TopKSampler()
    
    # 创建模拟预测 (2 个任务, 5 个候选)
    predictions = torch.tensor([
        [0.9, 0.5, 0.7, 0.3, 0.8],
        [0.85, 0.6, 0.65, 0.4, 0.75],
    ])
    candidate_ids = ["item_0", "item_1", "item_2", "item_3", "item_4"]
    
    results = sampler.sample(predictions, candidate_ids, top_k=3)
    
    assert len(results) == 3
    assert all(len(r) == 3 for r in results)  # (id, score, task_scores)
    print(f"  ✓ Top-K 采样正常: top_3={[r[0] for r in results]}")
    
    print("✅ Top-K 采样器测试通过")


def test_hstu_model_wrapper():
    """测试 HSTUModelWrapper 创建和前向传播"""
    print("\n=== 测试: HSTU 模型封装器 ===")
    
    from vllm_gr.config.hstu_config import HSTUModelConfig
    from vllm_gr.model_executor.embedding_manager import EmbeddingManager
    from vllm_gr.model_executor.models.hstu import HSTUModelWrapper
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用小型配置进行测试
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
    print(f"  ✓ HSTUModelWrapper 创建成功，设备: {device}")
    
    # 创建模拟输入
    seq_len = 5
    num_candidates = 3
    
    uih_embeddings = torch.randn(seq_len, config.embedding_table_dim, device=device)
    uih_lengths = torch.tensor([seq_len], dtype=torch.long, device=device)
    candidate_embeddings = torch.randn(num_candidates, config.embedding_table_dim, device=device)
    num_candidates_tensor = torch.tensor([num_candidates], dtype=torch.long, device=device)
    
    # 前向传播
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
    print(f"  ✓ 前向传播正常: predictions shape={output.predictions.shape}")
    
    print("✅ HSTU 模型封装器测试通过")


def test_grllm_basic():
    """测试 GRLLM 基本初始化和推荐功能"""
    print("\n=== 测试: GRLLM 基础功能 ===")
    
    from vllm_gr.entrypoint.gr import GRLLM
    from vllm_gr.config.hstu_config import HSTUModelConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用最小配置进行快速测试
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
    
    # 创建 GRLLM（无预训练权重）
    llm = GRLLM(model_path=None, config=config, device=device)
    print(f"  ✓ GRLLM 创建成功，设备: {device}")
    
    # 准备请求
    user_id = "user_001"
    sequence = [
        {"item_id": "movie_123", "timestamp": 1640995200},
        {"item_id": "movie_456", "timestamp": 1640995300},
        {"item_id": "movie_789", "timestamp": 1640995400},
    ]
    candidates = [{"item_id": f"candidate_{i}"} for i in range(10)]
    
    # 生成推荐
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
    print(f"  ✓ 推荐生成成功，耗时: {elapsed:.2f}ms (报告: {response.latency_ms:.2f}ms)")
    print(f"  ✓ Top 5 推荐: {[r['item_id'] for r in response.recommendations[:3]]}...")
    
    for i, rec in enumerate(response.recommendations):
        assert rec["rank"] == i
        assert "item_id" in rec
        assert "score" in rec
    
    print("✅ GRLLM 基础功能测试通过")


def test_grllm_batch():
    """测试 GRLLM 批量推荐"""
    print("\n=== 测试: GRLLM 批量推荐 ===")
    
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
    
    # 准备批量请求
    requests = [
        {
            "user_id": f"user_{i}",
            "sequence": [{"item_id": f"item_{j}", "timestamp": 1000 + j} for j in range(5)],
            "candidates": [{"item_id": f"cand_{k}"} for k in range(8)],
        }
        for i in range(3)
    ]
    
    # 批量推荐
    start_time = time.time()
    responses = llm.recommend_batch(requests, top_k=5)
    elapsed = (time.time() - start_time) * 1000
    
    assert len(responses) == 3
    for response in responses:
        assert len(response.recommendations) == 5
    
    print(f"  ✓ 批量推荐完成: 3 个用户，耗时 {elapsed:.2f}ms")
    print("✅ GRLLM 批量推荐测试通过")


def test_full_e2e_pipeline():
    """完整端到端测试，模拟真实使用场景"""
    print("\n=== 测试: 完整端到端流程 ===")
    
    from vllm_gr import GRLLM
    from vllm_gr.config.hstu_config import HSTUModelConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 创建配置
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
    print("  ✓ 步骤 1: 配置创建完成")
    
    # 2. 初始化 GRLLM
    llm = GRLLM(config=config, device=device)
    print(f"  ✓ 步骤 2: GRLLM 初始化完成，设备: {device}")
    
    # 3. 模拟真实请求
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
    
    # 4. 获取推荐
    response = llm.recommend(
        user_id=user_id,
        sequence=sequence,
        candidates=candidates,
        top_k=5,
        request_id="test_request_001",
    )
    print(f"  ✓ 步骤 3: 推荐生成完成")
    
    # 5. 验证响应
    assert response.request_id == "test_request_001"
    assert len(response.recommendations) == 5
    assert all("item_id" in r and "score" in r and "rank" in r for r in response.recommendations)
    
    print(f"\n  📊 结果:")
    print(f"     请求 ID: {response.request_id}")
    print(f"     延迟: {response.latency_ms:.2f}ms")
    print(f"     Top 5 推荐:")
    for rec in response.recommendations:
        print(f"       #{rec['rank']}: {rec['item_id']} (分数: {rec['score']:.4f})")
    
    print("\n✅ 完整端到端流程测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print(" vllm-GR 离线推理端到端测试套件")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  设备: {device}")
    print(f"🔧 PyTorch 版本: {torch.__version__}")
    
    tests = [
        ("配置创建", test_config_creation),
        ("嵌入管理器", test_embedding_manager),
        ("输入处理器", test_input_processor),
        ("输出处理器", test_output_processor),
        ("Top-K 采样器", test_topk_sampler),
        ("HSTU 模型封装器", test_hstu_model_wrapper),
        ("GRLLM 基础功能", test_grllm_basic),
        ("GRLLM 批量推荐", test_grllm_batch),
        ("完整端到端流程", test_full_e2e_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        return False
    else:
        print("\n🎉 所有测试通过！")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

