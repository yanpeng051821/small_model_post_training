# Week 7：GRPO 正式缩放复现

状态：未开始
前置门：Week 6 smoke 通过，正式配置可由证据解释。

## 1. 本周目标

从固定 B1 checkpoint 启动一次正式 GRPO run，得到 B2 checkpoint 和完整稳定性证据。

## 2. W7-1：冻结正式配置

从 `configs/grpo_smoke.yaml` 派生 `configs/grpo_train.yaml`。启动前保存配置 SHA256，并核对：

- B1 checkpoint 路径/hash 正确。
- 新 optimizer、新 scheduler、新输出目录，不恢复 SFT optimizer。
- 数据 manifest、reward 函数/权重、template/system prompt 未变化。
- G、生成参数、beta、epsilon、loss type、seed 明确。
- 与 smoke 和官方 recipe 的差异已经写入 revision note。

## 3. W7-2：执行正式 GRPO

```bash
cd "$OPENR1_REPO"
accelerate launch --num_processes 1 src/open_r1/grpo.py \
  --config "$OPENR1_BASELINE_ROOT/configs/grpo_train.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/grpo_train.log"
```

运行期间持续保存：

- 各 reward mean/std。
- total reward 与 zero-std group 比例。
- completion mean/max length、EOS 与 clipped ratio。
- loss、KL、clip ratio、grad norm。
- global step、checkpoint、任何中断与恢复。

## 4. 停止规则

以下情况先停止并保留最近 checkpoint/log，而不是临时改 reward：

- NaN/Inf。
- 大量 accuracy 无法解析。
- zero-std groups 长期占主导且无学习信号。
- completion 长度持续膨胀或 EOS/格式崩溃。
- reward 上升但 accuracy 不变、格式/长度单项异常，疑似 reward hacking。
- B2 输出写入 B1 目录或初始化模型不正确。

## 5. W7-3：B2 可用性检查

完成后加载 B2，运行与 B1 相同的 deterministic inference smoke，检查 template、EOS、格式和停止。保存 `artifacts/b2_checkpoint.json`，记录父 B1、配置 hash、数据 manifest、最终 step 和 checkpoint 路径。

## 6. 本周证据

```text
configs/grpo_train.yaml 与 SHA256
logs/grpo_train.log
B2 checkpoints / final model
trainer state / metrics
reward 与长度曲线
artifacts/b2_checkpoint.json
B2 inference smoke 输出
```

## 7. 完成标准

正式运行正常结束，或以有完整证据的 blocked 状态结束。仅有最终 checkpoint、没有配置和 rollout/reward 日志，不算可信 B2。
