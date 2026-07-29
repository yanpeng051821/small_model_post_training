# Week 6：GRPO/RLVR Smoke Test

状态：未开始
前置门：Week 5 明确批准 B1 进入 GRPO。

## 1. 本周目标

从 B1 出发，用官方 `grpo.py` 完成至少一次 prompt → G completions → rewards → advantage → policy loss → optimizer step → checkpoint 闭环。

## 2. W6-1：冻结 GRPO 数据与 reward

建立：

```text
data/grpo_manifest.json
data/grpo_sample_ids.jsonl
```

记录 `OpenR1-Math-220k/default/train` 的 revision/fingerprint、`shuffle(seed=42)`、前 N 条、`problem` 与 `solution` 字段。Smoke 使用 manifest 前 M 条。

Reward 固定为：

```text
accuracy + format + tag_count
```

把函数名、权重、Open-R1 commit 和正则格式写入 `configs/grpo_smoke.yaml`。不得在运行中根据短期表现改 reward。

## 3. W6-2：Reward 单元检查

在不更新模型的情况下，构造至少四类 completion：

- 正确且格式完整。
- 错误但格式完整。
- 正确但格式错误。
- 缺标签/被截断。

保存每个 reward 分项和总分，确认 `solution` 只进入 accuracy reward，`None` 与 `0.0` 没有混淆。输出：`data/reward_unit_audit.json`。

## 4. W6-3：运行最小 GRPO smoke

配置必须显式记录：B1 路径、数据 manifest、G、reward 权重、system prompt/template、max prompt/completion length、sampling、beta、epsilon、loss type、batch、gradient accumulation、seed、max steps 和输出目录。硬件/引擎参数由用户决定并原样保存。

```bash
cd "$OPENR1_REPO"
accelerate launch --num_processes 1 src/open_r1/grpo.py \
  --config "$OPENR1_BASELINE_ROOT/configs/grpo_smoke.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/grpo_smoke.log"
```

Smoke 目标是 1-2 个真实 optimizer steps，不是 reward 上升。

## 5. W6-4：检查核心信号

至少保存一个 prompt 的完整调试包：

```text
prompt 文本/token 数
G 个 completion 文本/token 数/EOS
accuracy、format、tag_count、total reward
group mean/std
advantage
zero-std 标记
policy loss、KL、clip ratio
```

通过条件：同题生成 G 个回答；reward 与样本对应；advantage 符合组内计算；有效 completion token 参与 loss；完成 optimizer step；checkpoint 可重载。

若所有组长期 zero-std、所有 accuracy 都是 `None`、格式 reward 被模板系统性破坏或 loss NaN，记录为 smoke 失败，先修复原因，不进入正式 GRPO。

## 6. 本周证据

```text
data/grpo_manifest.json
data/grpo_sample_ids.jsonl
data/reward_unit_audit.json
configs/grpo_smoke.yaml
logs/grpo_smoke.log
一个完整 prompt/group 调试包
smoke checkpoint 与 reload 输出
```

## 7. 完成标准

- [ ] rollout-to-update 闭环成功。
- [ ] reward、advantage、loss 能与源码路径对应。
- [ ] 正式 GRPO 配置的可用参数已经由 smoke 证据支持。
- [ ] 未为了“跑通”静默改变 B1、数据或 reward 语义。
