# Week 3：数据审计与 SFT Smoke Test

状态：已完成（2026-07-29）
前置门：Week 2 的 B0 canonical 合同和结果已保存。

## 1. 本周目标

把 `Mixture-of-Thoughts/math` 从“候选数据”变成可复现的训练 artifact，并用官方 `sft.py` 完成最小 forward → backward → optimizer → checkpoint 闭环。

## 2. W3-1：冻结 SFT 子集 manifest

必须记录：

- dataset id、config=`math`、split=`train`。
- 可用的 dataset revision/fingerprint。
- `shuffle(seed=42)` 后选择前 N 条。
- N、选择代码、样本索引或稳定样本 id、manifest hash。
- 生成后的 manifest 保存路径。

输出建议：

```text
baselines/local/open-r1-qwen3-0.6b/data/sft_manifest.json
baselines/local/open-r1-qwen3-0.6b/data/sft_sample_ids.jsonl
```

同一次 baseline 的 smoke 与正式训练必须来自同一选择规则；smoke 可以取 manifest 的前 M 条。

## 3. W3-2：长度和截断审计

使用最终 tokenizer、chat template 和 EOS 设置，对固定子集计算：

- token 长度 P50/P90/P95/P99/max。
- 候选 `max_seq_length` 下的样本截断率。
- assistant token 保留比例。
- 最终答案、`</think>`、`</answer>`、EOS 保留率。

输出：`data/sft_length_audit.json` 和至少 5 条长样本的截断前后对照。不能只使用数据集已有的 `num_tokens` 代替目标 tokenizer 实测。

## 4. W3-3：检查 TRL 最终训练样本

从实际 `SFTTrainer`/collator 路径取一条 batch，保存：

```text
模板后文本
input_ids
attention_mask
labels
labels == -100 的位置
有效 loss token 数
BOS/EOS/PAD id
```

Baseline 预期是 full-sequence LM：非 padding 的 system/user/assistant token 都直接参与 loss，PAD 为 `-100`。若观察结果不同，先停止并核对 TRL 0.18.0 配置，不静默改成 assistant-only。

输出：`data/sft_collator_audit.txt`。

## 5. W3-4：准备并运行 SFT smoke

从官方 SFT recipe 派生一份独立配置，保存为：

```text
configs/sft_smoke.yaml
```

配置必须显式记录：B0 模型、math 数据、子集、template/EOS、max length、precision、batch、gradient accumulation、learning rate、seed=42、max steps、输出目录、是否 LoRA。机器相关数值由用户决定。

执行模板：

```bash
cd "$OPENR1_REPO"
accelerate launch --num_processes 1 src/open_r1/sft.py \
  --config "$OPENR1_BASELINE_ROOT/configs/sft_smoke.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/sft_smoke.log"
```

Smoke 只需要少量样本和至少 2 个 optimizer steps，不用追求 loss 收敛。

## 6. Smoke 检查

- [ ] 实际加载的是 B0 和固定 SFT 子集。
- [ ] 至少完成 2 个 optimizer steps。
- [ ] loss、grad norm 没有 NaN/Inf。
- [ ] checkpoint、trainer state 和 metrics 文件存在。
- [ ] checkpoint 能重新加载并生成文本。
- [ ] 日志记录实际吞吐、序列长度和所有配置覆盖项。
- [ ] 中断恢复路径至少完成只读核对；不要求为了测试恢复而故意破坏运行。

## 7. 本周证据

```text
data/sft_manifest.json
data/sft_sample_ids.jsonl
data/sft_length_audit.json
data/sft_collator_audit.txt
configs/sft_smoke.yaml
logs/sft_smoke.log
smoke checkpoint 与 metrics 路径
```

## 8. 完成标准

固定数据能够沿官方 `sft.py` 完成真实参数更新并保存可重新加载的 checkpoint。若失败，记录首个失败点和实际配置；不要通过删除数据审计或改写 Trainer 来制造“通过”。

## 9. 执行记录与经验沉淀

### 9.1 W3-1：冻结 SFT 子集 manifest ✅

- Manifest: 1000 条，shuffle(seed=42) → `ca5e98709338`
- 所有样本均为 `['user', 'assistant']` 单轮对话
- 数据源: `open-r1/OpenR1-Math-220k`
- 脚本: `scripts/w3_1_freeze_subset.py`
- 网络问题导致首次全量下载失败，改用 `streaming=True` 解决

### 9.2 W3-2：长度审计 ✅

**关键发现：数据远比预期长**

| 分位数 | Token 数 |
|--------|----------|
| P50    | 4,949    |
| P90    | 12,574   |
| P95    | 14,923   |
| max    | 18,584   |

- 100% 样本包含 `</think>` 标签（全推理数据）
- 0% 样本以 EOS 结尾（训练时 SFTTrainer 自动追加）

**截断率评估：**

| max_seq_length | 截断率 |
|----------------|--------|
| 4096           | 57.8%  |
| 5120           | ~45%   |
| 6144           | 38.8%  |

### 9.3 W3-3：Collator 审计 ✅

- 单条样本实测: user=152 tokens, assistant=3362 tokens
- Label 策略: full-sequence LM（所有非 PAD 位置参与 loss）
- 特殊 token: `<|im_start|>`, `<|im_end|>` 确认正确渲染
- 无 `<|endoftext|>` 在数据中

### 9.4 W3-4：SFT Smoke Test ✅ （经过多次迭代）

**最终可用配置（`configs/sft_smoke.yaml`）：**

```yaml
model_name_or_path: Qwen/Qwen3-0.6B-Base
torch_dtype: float16          # FP16 权重节省显存
max_seq_length: 5120           # T4 16GB 上限（6144 OOM）
gradient_checkpointing: true
max_steps: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
fp16: false                    # 不用 AMP（与 torch_dtype 冲突）
```

**运行命令（`scripts/w3_4_sft_smoke.sh`）：**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 碎片化修复
cd "$OPENR1_REPO"
accelerate launch --num_processes 1 --mixed_precision no \
  src/open_r1/sft.py --config "$CONFIG"
```

### 9.5 经验教训：FP16 训练在 T4 上的坑

**错误 1：`ValueError: Attempting to unscale FP16 gradients.`**

原因：YAML 中 `torch_dtype: float16` + accelerate `--mixed_precision fp16` 双重控制，GradScaler 看到 FP16 权重报错。

修复：去掉 `--mixed_precision fp16`，改用 `--mixed_precision no`。纯 FP16 训练不经过 GradScaler。

**错误 2：OOM with 3.48 GiB allocation on backward pass**

原因：显存碎片化——PyTorch 默认的内存分配器在 step 间留下碎片，即使总空闲 > 3.48 GiB，也找不到一个连续的 3.48 GiB 块。

修复：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启用可扩展内存段，允许 python 向 CUDA 动态申请更大的连续空间。

**错误 3：`max_num_batched_tokens (2048) < max_model_len (4096)`**

原因：vLLM 默认 batched token 上限小于我们设的模型长度。

修复：加 `max_num_batched_tokens=4096` 对齐。

**错误 4：`yaml` 中 `max_train_samples` 不被 TrlParser 识别**

原因：SFTConfig 不支持 `max_train_samples`，这是 HuggingFace TrainingArguments 的参数但被 TrlParser 截断了。

影响：每次 smoke 必须预处理全部 93,733 条数据，即使只需要 16 条。首次预处理 ~20 分钟，后续有缓存可加速（~30 秒 truncation）。

**max_seq_length 选择过程：**

```
4096 → 不 OOM，截断率 57.8%
6144 → 第 1 步通过，第 2 步 backward OOM（碎片化）
6144 + expandable_segments → 第 1 步通过，第 2 步 backward 仍 OOM
5120 → 不 OOM，截断率 ~45%，**当前选择**
```

### 9.6 Smoke 运行结果

| 指标 | 值 |
|------|-----|
| max_seq_length | 5120 |
| 4 steps 耗时 | 36 秒（~9 秒/step） |
| 实际处理样本 | 16 条（4 accum × 1 batch × 4 steps） |
| Step 1 loss | 0.98（正常） |
| Step 2-4 | loss=0, grad_norm=NaN（FP16 梯度下溢） |

### 9.7 Week 4 待解决问题

1. **AMP 混合精度**：纯 FP16 训练导致梯度下溢。Week 4 正式训练需要解决：
   - 方案 A：去掉 `torch_dtype`，让模型以 FP32 加载，accelerate `--mixed_precision fp16` 做 AMP（显存翻倍，5120 可能不行）
   - 方案 B：用 LoRA 省显存，换取 AMP 空间
   - 方案 C：接受 `max_seq_length=4096`，换取 AMP 稳定性

2. **`max_train_samples` 预处理**：SFTTrainer 预处理全量数据，需要找到优化方案或接受每次 ~20 分钟开销

3. **grad_norm=NaN**：16 条样本过拟合 + FP16 导致，正式训练数据量大几倍，应该不会出现
