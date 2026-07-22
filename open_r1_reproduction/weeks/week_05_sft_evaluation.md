# Week 5：B1 评测、失败分析与 GRPO 进入门

状态：未开始
前置门：B1 checkpoint、训练日志和配置完整。

## 1. 本周目标

使用 Week 2 冻结的同一 LightEval 合同比较 B0/B1，判断 SFT 改变了什么，以及 B1 是否适合作为 GRPO 起点。

## 2. W5-1：同条件评测

B0 已有 canonical 结果；对 B1 运行完全相同的：

```text
MATH-500 task
chat template / system prompt
EOS/PAD
temperature / top-p / max tokens
每题响应数与 seed policy
答案提取与指标聚合
```

输出到 `evals/b1-canonical/`，保存实际命令、汇总结果和逐题 details。若合同发生任何变化，必须同时重跑 B0。

## 3. W5-2：B0/B1 对照

至少生成以下表：

| 指标 | B0 | B1 | Delta |
|---|---:|---:|---:|
| MATH-500 pass@1/accuracy | | | |
| format rate | | | |
| EOS rate | | | |
| 平均输出 token | | | |
| 截断率 | | | |

训练 reward、LightEval accuracy 和格式指标分开报告，不用一个指标替代另一个。

## 4. W5-3：样本级失败分析

按逐题输出分类并各保存代表样本：

- B0 错 → B1 对。
- B0 对 → B1 错。
- 都对但 B1 更长/更短。
- 格式正确但答案错误。
- 未闭合标签、无 EOS、截断或无法解析。

输出 `analysis/b0_b1_error_analysis.md`，至少包含计数、代表样本和可能原因；不要只摘取有利案例。

## 5. W5-4：GRPO 进入决定

可以进入 Week 6 的条件：

- B1 可稳定加载、生成和停止。
- 没有无法解释的灾难性退化。
- 输出格式和初始正确率足以让 reward 产生非零、非恒定信号。
- B0/B1 结果可比较且原始输出存在。

分数不必强制上涨；可信的负结果仍可保留。但如果数据/模板/评测不可比，应先 repair，不进入 GRPO。

## 6. 本周证据与完成标准

```text
evals/b1-canonical/ 汇总和 details
B1 实际评测命令
B0/B1 指标对照表
analysis/b0_b1_error_analysis.md
进入 GRPO、repair 或 stop 的书面决定
```

只有书面 gate 决定为 continue，Week 6 才开始。
