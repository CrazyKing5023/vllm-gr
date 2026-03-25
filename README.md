# vllm-GR

**vLLM-based Generative Recommender Inference Engine**

当前仓库聚焦于基于 vLLM 的生成式推荐推理，包含 HSTU 离线推理主链路、示例脚本、测试套件，以及 GPU / NPU 运行时适配代码。

## Inference Scope

当前实现是面向推理场景的 HSTU 适配，不追求完整训练态功能覆盖。为避免后续维护时产生误判，下面两点是当前仓库有意保留的范围边界：

- `vllm_gr/models/dlrm_v3/modules/stu.py` 仅保证批量打分路径，未恢复 golden 中完整的增量 KV cache / `cached_forward()` 语义。
- `vllm_gr/models/dlrm_v3/modules/multitask_module.py` 仅保留推理预测输出，不再计算训练所需的 labels / weights / losses。

如果后续目标从“推理适配”扩展到“训练或增量推理完全对齐”，需要把以上两块重新纳入 parity 范围。

## 仓库当前包含的内容

- `GRLLM`：面向调用方的推荐入口
- `GRPrompt`：推荐请求输入协议
- `GRRequestOutput` / `RecommendedItem`：推荐结果输出协议
- `vllm_gr/worker`、`vllm_gr/utils`、`vllm_gr/ops`：平台相关运行时边界
- `examples/`：可直接参考的推理示例
- `scripts/`：测试与 checkpoint 转换脚本
- `tests/`：单测、平台 smoke、清理与守护测试

## 目录概览

```text
vllm_gr/
examples/
scripts/
tests/
requirements-gpu.txt
requirements-npu.txt
pyproject.toml
```

## 安装

基础安装：

```bash
pip install .
```

开发安装：

```bash
pip install -e ".[dev]"
```

GPU 运行时：

```bash
pip install -r requirements-gpu.txt
# 或
pip install ".[gpu]"
```

NPU 运行时：

```bash
pip install -r requirements-npu.txt
# 或
pip install ".[npu]"
```

完整安装：

```bash
pip install ".[all-gpu]"
pip install ".[all-npu]"
```

`.[all]` 仍保留为兼容别名，默认对齐 GPU 路径。

## 快速开始

最直接的入口是仓内示例脚本：

```bash
python examples/simple_inference.py
python examples/batch_inference.py
python examples/dataset_inference.py debug
```

也可以直接调用 Python API：

```python
from vllm_gr import GRLLM, GRPrompt

llm = GRLLM(
    model="examples/demo-model",
    max_seq_len=64,
    max_candidates=50,
)

prompt: GRPrompt = {
    "user_id": "user_001",
    "user_history": [
        {"item_id": 1001, "timestamp": 1609459200},
        {"item_id": 1002, "timestamp": 1609545600},
    ],
    "candidates": [
        {"item_candidate_id": 2001, "query_time": 1610000000},
        {"item_candidate_id": 2002, "query_time": 1610000000},
    ],
}

outputs = llm.recommend([prompt], use_tqdm=False)
print(outputs[0].item_ids)
llm.shutdown()
```

`GRPrompt` 中 `user_history` / `candidates` 的特征名需要和模型配置保持一致。不同数据集的写法可以直接参考 `examples/` 与 `vllm_gr/inputs/data.py`。

## 常用脚本

运行测试：

```bash
python -m pytest tests -v
python scripts/run_tests.py
python scripts/run_tests.py --suite platform-gpu
python scripts/run_tests.py --suite platform-npu
```

转换训练 checkpoint：

```bash
python scripts/convert_gr_checkpoint.py --help
```

加载转换后的模型：

```bash
python examples/load_converted_model.py --help
```

卸载：

```bash
pip uninstall vllm-gr
```

## 补充文档

README 只保留仓内已有代码、脚本和直接入口。补充性的运行说明、治理说明和扩展参考已整理到：

- [docs/readme_supplement.md](docs/readme_supplement.md)
- [docs/self_validation_checklist.md](docs/self_validation_checklist.md)
- [docs/golden_e2e_compare.md](docs/golden_e2e_compare.md)
