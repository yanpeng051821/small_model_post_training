# Open-R1 八周运行与验证子路线

本文件只追踪 Open-R1 的服务器运行、B0/B1/B2 证据和同条件验证，是项目级主路线的 L0。后续课程、独立实现与论文复现以[项目级主路线](../small_model_post_training_research_and_roadmap.md)为准。

## 1. 总目标

使用 Open-R1 官方代码路径，在 Qwen3-0.6B-Base 上完成：

```text
B0 Base Eval
-> B1 Reasoning SFT
-> B1 Eval
-> B2 GRPO/RLVR
-> B2 Eval
-> Baseline Verification
```

## 2. 进度看板

| Week | 主题 | 状态 | 核心产出 |
|---|---|---|---|
| 1 | 源码地图与复现合同 | 已完成 | pinned source、entrypoint、数据与指标合同 |
| 2 | 环境、推理与 B0 baseline | 证据待补 | 已报告 B0 指标；需补结果摘要、实际命令和服务器证据指针 |
| 3 | 数据审计与 SFT smoke test | 证据待归档 | 审计完成；历史 4-step smoke 出现 NaN，需说明与后续 B1 正式训练的关系 |
| 4 | Reasoning SFT 缩放复现 | 已报告完成 | B1 训练完成；配置、日志、checkpoint 和可用性证据待同步 |
| 5 | SFT 评测与失败分析 | 进行中 | B1 LightEval 结果、B0/B1 对照和失败样本分类待归档 |
| 6 | GRPO/RLVR smoke test | 未开始 | rollout-reward-advantage-update 闭环 |
| 7 | GRPO 缩放复现 | 未开始 | B2 checkpoint、RL 日志与稳定性记录 |
| 8 | 最终验证与阶段决策 | 未开始 | B0/B1/B2 对照、verification、baseline verdict |

## 3. 每周入口

- [Week 1：源码地图与复现合同](weeks/week_01_source_and_contract.md)
- [Week 2：环境、推理与 B0 baseline](weeks/week_02_environment_and_baseline.md)
- [Week 3：数据审计与 SFT smoke test](weeks/week_03_data_and_sft_smoke.md)
- [Week 4：Reasoning SFT 缩放复现](weeks/week_04_sft_reproduction.md)
- [Week 5：SFT 评测与失败分析](weeks/week_05_sft_evaluation.md)
- [Week 6：GRPO/RLVR smoke test](weeks/week_06_grpo_smoke.md)
- [Week 7：GRPO 缩放复现](weeks/week_07_grpo_reproduction.md)
- [Week 8：最终验证与阶段决策](weeks/week_08_verification_and_handoff.md)

## 4. 节奏规则

- 本八周编号只服务 L0，不与后续项目阶段编号互相替代。
- 一次只推进一个 Week。
- 每周先完成核心任务，再考虑加餐。
- 训练前必须有 smoke test；smoke 失败不进入正式运行。
- 同一失败类型最多进行一次有依据的修复重试。
- 每周结束时填写产出、指标、阻塞和复盘。
- 任何结果只有经过 Week 8 同条件验证后才可称为可信 baseline。
