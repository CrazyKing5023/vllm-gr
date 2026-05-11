# HSTU 权重转换与裁剪使用说明

本文记录 `vllm-GR/scripts/convert_gr_checkpoint.py` 的用途、推荐使用方式、验收方法，以及本次接入过程中踩过的坑。

适用场景：

- 输入是 `generative-recommenders` 训练产物
- 需要把 native checkpoint 转成 `vllm-GR` 可加载格式
- 需要裁剪 embedding 表行数，减小产物体积
- 需要把转换产物拆成 `dense weights` 和 `embedding weights`

## 1. 输入与输出

### 1.1 native checkpoint 目录格式

`generative-recommenders` 的训练产物通常是：

```text
<checkpoint_dir>/
├── non_sparse.ckpt
└── sparse/
```

- `non_sparse.ckpt`：dense 部分
- `sparse/`：embedding 部分，使用 PyTorch Distributed Checkpoint 保存

### 1.2 转换后目录格式

若启用 `--split-weights --format safetensors`，输出通常为：

```text
<converted_model_dir>/
├── config.json
├── dense.safetensors
├── embedding.safetensors
└── conversion_meta.json
```

若不启用 `--split-weights`，则会输出单文件：

```text
<converted_model_dir>/
├── config.json
├── model.safetensors
└── conversion_meta.json
```

`conversion_meta.json` 会记录：

- 输入 checkpoint 路径
- 输出格式
- 是否 split
- embedding 裁剪维度/行数
- 各 embedding table 的最终 `num_embeddings`

## 2. 当前支持的能力

`scripts/convert_gr_checkpoint.py` 当前支持：

- native checkpoint -> `vllm-GR` / HuggingFace 风格权重
- `dense` 与 `embedding` 拆文件输出
- `--crop-num-embeddings`：裁剪 embedding 表行数
- `--crop-embedding-dim`：裁剪 embedding 维度
- `--dense-only`：只转换 dense，跳过 sparse
- `--format safetensors|pt`
- `--dtype bfloat16|float16|float32`

### 2.1 `--crop-num-embeddings` 的真实语义

这个参数裁剪的是 embedding 表的**行数**，不是维度。

例如：

```bash
--crop-num-embeddings 32768
```

表示：

- 每个 embedding table 只保留前 `32768` 行
- `config.json` 中对应 table 的 `num_embeddings` 也同步减小
- 超出范围的 ID 不会做 remap，按当前实现应直接 fail-fast

### 2.2 如果原始表比指定裁剪值更小

脚本行为是：

- 实际保留 `min(原始行数, crop_num_embeddings)` 行
- 不会补齐，不会扩表

例如原始表只有 `10000` 行，而你传 `--crop-num-embeddings 32768`：

- 最终仍保留 `10000` 行
- `config.json` 中也仍是 `10000`

## 3. 推荐使用方式

### 3.1 推荐命令：产出 bf16 版本

这是线上更常见的产物形式：

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path /path/to/native_checkpoint \
  --output-dir /path/to/converted_model \
  --dataset movielens-1m \
  --crop-num-embeddings 32768 \
  --split-weights \
  --format safetensors \
  --dtype bfloat16 \
  --verbose
```

说明：

- `--dataset movielens-1m`：让脚本生成正确的 full source config
- `--crop-num-embeddings 32768`：裁行数
- `--split-weights`：拆成 `dense.safetensors` + `embedding.safetensors`
- `--dtype bfloat16`：dense 权重按指定 dtype 输出

### 3.2 推荐命令：做精确对齐时产出 fp32 版本

做 golden 对比时，建议先出一版 `float32`，便于排除 dtype 干扰：

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path /path/to/native_checkpoint \
  --output-dir /path/to/converted_model_fp32 \
  --dataset movielens-1m \
  --crop-num-embeddings 32768 \
  --split-weights \
  --format safetensors \
  --dtype float32 \
  --verbose
```

## 4. 不同参数的使用建议

### 4.1 `--dataset` 与 `--config-json` 怎么选

优先建议：

- 已知数据集，用 `--dataset`
- 只有一份可靠的训练期 full config 时，用 `--config-json`

不建议：

- 把已经转换后的、已经裁剪过的 `config.json` 再拿回来当 source config

原因见“常见坑”。

### 4.2 什么时候用 `--split-weights`

推荐开启：

- 便于单独检查 dense 和 embedding 产物
- 便于定位 sparse/dense 哪一侧有问题
- 更符合本次接入的文件层拆分需求

### 4.3 什么时候用 `--dense-only`

仅适用于：

- 你只想导出 dense 权重
- embedding 走外部特征服务

如果同时需要 `--crop-num-embeddings` 或 `--crop-embedding-dim`，不要配 `--dense-only`。

## 5. 推荐验收流程

推荐使用 `generative-recommenders` 作为 golden，对同一批输入逐 case 对比分数。

### 5.1 生成测试输入

输入中的所有 embedding ID 必须落在裁剪范围内。

例如裁到 `32768` 行时：

```bash
python "$GOLDEN_REPO/generative_recommenders/dlrm_v3/inference/e2e_test/acc_datamaker.py" \
  --output-json "$INPUT_JSON" \
  --num-cases 200 \
  --seed 20260318 \
  --item-id-max 32767 \
  --context-id-max 32767
```

### 5.2 生成 golden GT

```bash
export PYTHONPATH="$GOLDEN_REPO:$PYTHONPATH"

python "$GOLDEN_REPO/generative_recommenders/dlrm_v3/inference/e2e_test/acc_output.py" \
  --model-path "$NATIVE_CKPT" \
  --input-json "$INPUT_JSON" \
  --output-json "$GT_JSON" \
  --dataset movielens-1m
```

### 5.3 跑 `vllm-GR`

```bash
python examples/acc_test.py \
  --model "$CONVERTED_MODEL" \
  --input-json "$INPUT_JSON" \
  --output-json "$PRED_JSON" \
  --gt-json "$GT_JSON"
```

### 5.4 分析误差

```bash
python examples/analyze_acc_error.py \
  --pred-json "$PRED_JSON" \
  --gt-json "$GT_JSON" \
  --input-json "$INPUT_JSON"
```

### 5.5 出现大误差时怎么继续排查

若 `mean_absolute_error` 很大，优先跑：

```bash
python examples/compare_sparse_case.py \
  --native-model "$NATIVE_CKPT" \
  --converted-model "$CONVERTED_MODEL" \
  --input-json "$INPUT_JSON" \
  --case-id case_000003 \
  --golden-repo "$GOLDEN_REPO" \
  --dataset movielens-1m
```

它会帮助判断：

- touched sparse rows 是否一致
- seq_embeddings 是否一致
- payload_features 是否一致

如果 sparse 侧全部对齐，而最终误差仍很大，通常说明问题在 dense 侧或 config 语义侧。

## 6. 本次确认过的关键坑

### 6.1 坑一：`--crop-num-embeddings` 裁的是行数，不是维度

本次一开始最容易混淆的是：

- `crop-num-embeddings`：裁 embedding 表条目数
- `crop-embedding-dim`：裁 embedding 向量维度

如果目标是“减少 embedding 表里保留的条目数”，应使用：

```bash
--crop-num-embeddings
```

### 6.2 坑二：不能把“已经裁剪后的 config.json”再当 source config

错误用法：

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path "$NATIVE_CKPT" \
  --output-dir "$OUT" \
  --config-json "$OLD_CONVERTED_MODEL/config.json" \
  --crop-num-embeddings 32768
```

如果这个 `config.json` 里已经是裁剪后的：

- `movie_id.num_embeddings = 32768`

那脚本在加载原始 sparse checkpoint 时，会用一个 `32768` 行的模型去读原始 `10000000` 行的 embedding 表，最终报：

- saved shape 与 current shape 不匹配

正确做法：

- 用 `--dataset movielens-1m`
- 或传训练期的 full source config

### 6.3 坑三：不要直接拿 `examples/configs/movielens-1m/config.json` 当 full source config

仓里的示例 config 主要用于 demo / 示例，不一定等于训练期 full table 大小。

本次确认过：

- 某些示例 config 中 `num_embeddings` 是小值
- 真实训练/golden 配置中 `movielens-1m` 的 table 常常是 `HASH_SIZE = 10000000`

因此，转换原始 sparse checkpoint 时，优先使用：

```bash
--dataset movielens-1m
```

### 6.4 坑四：`config.json` 中 contextual dict 的 key 顺序会影响 dense 推理结果

这是本次定位出的核心坑。

HSTU dense 路径会按：

- `contextual_feature_to_max_length.keys()`
- `contextual_feature_to_min_uih_length.keys()`

的迭代顺序去拼接 contextual features。

如果训练时顺序是：

```python
["user_id", "sex", "age_group", "occupation", "zip_code"]
```

但转换后 `config.json` 变成：

```python
["age_group", "occupation", "sex", "user_id", "zip_code"]
```

那么即使：

- sparse embedding 行内容完全正确
- seq_embeddings 完全一致

最终 dense 分数仍会大幅错位。

本次已经在 `convert_gr_checkpoint.py` 中修复：

- 写出目标 `config.json` 时，显式按 source config 的 contextual key 顺序保序落盘

因此当前版本**不需要再手工修改 `config.json` 顺序**。

### 6.5 坑五：golden `acc_output.py` 的 import 可能失败

常见报错：

```text
No module named 'generative_recommenders'
```

建议：

```bash
export GOLDEN_REPO=/path/to/generative-recommenders
export PYTHONPATH="$GOLDEN_REPO:$PYTHONPATH"
```

然后再运行 `acc_output.py`。

### 6.6 坑六：环境里 `torchao` / `fbgemm_gpu.experimental` 可能导致 `GRLLM` import 失败

如果遇到类似：

```text
ModuleNotFoundError: No module named 'fbgemm_gpu.experimental'
```

说明当前 Python 环境里某些可选依赖版本不一致，需要先对齐环境，再跑 `GRLLM` 路径。

### 6.7 坑七：只有相对误差大，不一定代表真有大问题

当 GT 分数本身极接近 `0` 时：

- 非常小的绝对误差
- 也可能表现成很高的相对误差

因此更推荐同时看：

- `mean_absolute_error`
- `p95_absolute_error`
- `max_absolute_error`

如果像本次修复后这样：

- `mean_absolute_error` 非常小
- `p95_absolute_error` 很小

那通常已经说明结果是对齐的。

## 7. 推荐排障顺序

推荐按下面顺序排：

1. 先确认输入 ID 都落在裁剪范围内。
2. 再确认 source config 是 full config，不是裁剪后 config。
3. 用 `compare_sparse_case.py` 看 sparse rows / seq_embeddings / payload_features。
4. 如果 sparse 全对齐，但 acc 仍差，优先检查 `config.json` 的 contextual key 顺序。
5. 最后再看 dtype 问题。

## 8. 一组可复用命令模板

### 8.1 产出 bf16 裁剪模型

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path "$NATIVE_CKPT" \
  --output-dir "$CONVERTED_MODEL" \
  --dataset movielens-1m \
  --crop-num-embeddings 32768 \
  --split-weights \
  --format safetensors \
  --dtype bfloat16 \
  --verbose
```

### 8.2 产出 fp32 验证模型

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path "$NATIVE_CKPT" \
  --output-dir "$CONVERTED_MODEL_FP32" \
  --dataset movielens-1m \
  --crop-num-embeddings 32768 \
  --split-weights \
  --format safetensors \
  --dtype float32 \
  --verbose
```

### 8.3 误差分析

```bash
python examples/analyze_acc_error.py \
  --pred-json "$PRED_JSON" \
  --gt-json "$GT_JSON" \
  --input-json "$INPUT_JSON"
```

## 9. 当前结论

本次功能已经具备以下能力：

- 支持把 native checkpoint 转成 `vllm-GR` 可加载格式
- 支持裁剪 embedding 行数
- 支持拆分 `dense weights` / `embedding weights`
- 支持使用 golden 链路做逐 case 精度对比
- 已修复 `config.json` contextual key 顺序导致的 dense 错位问题

如果后续继续扩展，建议优先保持：

- source config 与 target config 的语义分离
- 配置落盘顺序稳定
- 验收流程固定走 `acc_datamaker.py` + `acc_output.py` + `acc_test.py` + `analyze_acc_error.py`
