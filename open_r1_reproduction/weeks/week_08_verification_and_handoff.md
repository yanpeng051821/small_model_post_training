# Week 8：B0/B1/B2 最终验证与交接

状态：未开始
前置门：B0、B1、B2 均存在，或 B2 已有明确 blocked 证据。

## 1. 本周目标

用同一 canonical 合同完成最终比较，形成可供后续自定义实验复用的 baseline 结论，而不是只报告“代码跑通”。

## 2. W8-1：可比性审计

逐项核对 B0/B1/B2：

- source commit、实际依赖和模型 lineage。
- dataset revision/fingerprint、split、manifest 与 seed。
- chat template、system prompt、EOS/PAD。
- LightEval 版本、task、生成参数、响应数、seed policy。
- 答案提取和 metric aggregation。

任何一项不同都必须解释影响；影响指标语义时先重跑对应模型，不能直接比较。

## 3. W8-2：最终评测

对 B0/B1/B2 使用相同 LightEval 命令模板，分别输出到：

```text
evals/final/b0/
evals/final/b1/
evals/final/b2/
```

保存汇总结果、逐题 details、实际命令和运行日志。若 AIME 预算允许，作为次指标执行；MATH-500 是最低必需指标。

## 4. W8-3：最终结果表

至少生成：

| 指标 | B0 | B1 | B2 | B1-B0 | B2-B1 |
|---|---:|---:|---:|---:|---:|
| MATH-500 pass@1/accuracy | | | | | |
| format rate | | | | | |
| EOS rate | | | | | |
| 平均输出 token | | | | | |
| 截断率 | | | | | |

训练指标另表记录：SFT loss、GRPO reward 分项、zero-std、KL、clip ratio。不要把训练 reward 与 benchmark accuracy 放在同一数值含义下比较。

## 5. W8-4：样本与失败分析

对 B0→B1、B1→B2 分别统计：错→对、对→错、格式变化、长度变化、无法解析、截断和疑似 reward hacking。每类保留代表样本，并区分：

- 实现/配置偏差。
- 数据或 split 偏差。
- 评测合同偏差。
- 随机采样波动。
- 模型真实行为变化。

## 6. W8-5：写 verification.md

报告必须自包含：

```text
baseline id / variant
源码和模型 lineage
数据与评测合同
实际命令和配置路径
可信指标与逐题输出路径
B0/B1/B2 结论
已知偏差和限制
失败分析
可复用程度
下一阶段建议
```

结果分类：`verified_match`、`verified_close`、`verified_diverged` 或 `broken`。信任分类：`verified`、`partially_verified`、`operational_but_incomparable` 或 `failed`。

## 7. W8-6：阶段决策

- 可比较且证据完整：接受 baseline，进入 TinyTutor/自定义消融。
- 有可修复的单点错误：进入 repair，保留当前结果，不覆盖。
- 只能运行但不可比较：标记 `operational_but_incomparable`。
- 资源或核心路径不可行：记录 blocked 和重开条件。

## 8. 最终证据与完成标准

```text
evals/final/b0|b1|b2 汇总与 details
最终对照表
最终失败分析
verification.md
所有配置/manifest/checkpoint/log 指针
父目录 PROGRESS.md 更新
baseline accept/repair/blocked 决定
```

Baseline 完成的标准是“结论可信且可复查”，不是强制要求 B2 分数高于 B1。
