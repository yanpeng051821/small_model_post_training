# Week 3：数据审计与 SFT Smoke Test

状态：未开始
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
