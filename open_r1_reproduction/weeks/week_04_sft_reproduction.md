# Week 4：Reasoning SFT 正式缩放复现

状态：B1 正式训练已报告完成；配置、日志、checkpoint 与可用性证据待同步
证据门：必须说明历史 Week 3 NaN 与本次正式训练的关系，并证明训练输入与冻结数据合同一致。

## 1. 本周目标

使用官方 `sft.py` 完成一次预先冻结的正式运行，得到 B1 checkpoint。该周不做无计划超参搜索。

## 2. W4-1：冻结正式配置

从 `configs/sft_smoke.yaml` 派生：

```text
configs/sft_train.yaml
```

只允许基于 smoke 证据调整训练规模、batch/accumulation、长度、checkpoint 频率和运行时参数。模型、数据语义、full-sequence loss、template/EOS 和 seed 不得静默变化。

启动前保存：

- 完整 YAML。
- 配置文件 SHA256。
- Open-R1 commit。
- 数据 manifest hash。
- 预计 steps/epochs 和输出目录。
- 与官方 recipe、smoke 配置的差异表。

## 3. W4-2：执行正式 SFT

```bash
cd "$OPENR1_REPO"
accelerate launch --num_processes 1 src/open_r1/sft.py \
  --config "$OPENR1_BASELINE_ROOT/configs/sft_train.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/sft_train.log"
```

运行期间记录：global step、loss、learning rate、grad norm、tokens/s、step time、checkpoint 和任何重启。硬件指标可由用户按需要记录，但不是本路线的判断职责。

## 4. 停止与恢复规则

立即停止并保存证据：NaN/Inf、数据字段变化、错误 checkpoint、输出目录覆盖 B0、loss mask 与 Week 3 不一致。普通波动不是调参理由。

发生中断时只从 Trainer 保存的 checkpoint 恢复，并记录原命令、恢复命令和 checkpoint；不能重新开始后仍声称是同一次连续运行。

## 5. W4-3：B1 可用性检查

训练结束后：

- 加载最终 B1 checkpoint 与 tokenizer。
- 使用 Week 2 相同的 2+3 prompt 做 deterministic inference smoke。
- 检查 EOS、格式、生成停止和 checkpoint 文件完整性。
- 保存 `artifacts/b1_checkpoint.json`，记录路径、来源 B0、配置 hash、manifest hash、最终 step。

## 6. 本周证据

```text
configs/sft_train.yaml
configs/sft_train.sha256
logs/sft_train.log
B1 checkpoints / final model
train metrics / trainer state
artifacts/b1_checkpoint.json
B1 inference smoke 输出
```

## 7. 完成标准

- [ ] 正式运行按冻结配置正常结束或从有证据的 checkpoint 恢复结束。
- [ ] B1 可独立加载和生成。
- [ ] 配置、数据、源码、日志和 checkpoint 能互相定位。
- [ ] 尚未用训练 loss 宣称 benchmark 提升；能力结论留到 Week 5。
