# 小参数模型后训练专项：总进度

最后更新：2026-07-27

## 1. 当前状态

| 项目 | 当前值 |
|---|---|
| 专项目标 | 掌握并验证小参数模型的完整后训练流程 |
| 当前主线 | Open-R1 缩放复现 |
| 基础模型 | Qwen3-0.6B-Base |
| 当前阶段 | SFT Smoke & Reproduction |
| 当前周 | Week 3：数据审计与 SFT smoke test |
| Baseline 信任状态 | B0 acquired (MATH-500: 26-28%) |
| 当前是否允许自定义实验 | 否；先通过 Open-R1 baseline gate |
| B0 评测合同 | temperature=0.6, max_new_tokens=512, max_model_length=4096, chat_template=Qwen3 default, 1n+4n pass@1 |

Week 1-2 已完成：源码固定、环境审计、tokenizer/推理验证、数据审计、vLLM+LightEval 链路验证、B0 canonical eval 完成（MATH-500 全部 500 题）。评测合同已冻结。Week 3 进入 SFT 数据审计和最小训练闭环。

## 2. 大阶段看板

| 阶段 | 内容 | 状态 | 完成条件 |
|---|---|---|---|
| P0 | 行业、实验室和招聘方向调研 | 已完成 | 研究方向与岗位能力地图已形成 |
| P1 | 选择可信开源基线 | 已完成 | 选定 Open-R1 + Qwen3-0.6B-Base |
| P2 | Open-R1 缩放复现 | 进行中 | B0/B1/B2 可复现且完成同条件评测 |
| P3 | TinyTutor 教育场景迁移 | 未开始 | 教育数据、reward 与 eval contract 固定 |
| P4 | 后训练方法对照实验 | 未开始 | 至少三种方法完成公平比较 |
| P5 | 作品集与技术报告 | 未开始 | 代码、数据、指标、失败分析可复现 |

## 3. 当前活跃子项目

- 子项目入口：[`open_r1_reproduction/README.md`](open_r1_reproduction/README.md)
- 八周路线：[`open_r1_reproduction/open_r1_8_week_roadmap.md`](open_r1_reproduction/open_r1_8_week_roadmap.md)
- Baseline 计划：[`open_r1_reproduction/PLAN.md`](open_r1_reproduction/PLAN.md)
- Baseline 检查清单：[`open_r1_reproduction/CHECKLIST.md`](open_r1_reproduction/CHECKLIST.md)

## 4. 阶段门

进入 TinyTutor 自定义实验前，Open-R1 子项目必须满足：

- [x] 固定上游源码 commit 和环境版本。
- [x] 获得 Qwen3-0.6B-Base 训练前 baseline。（B0: MATH-500 ≈ 27%）
- [ ] 使用官方 `sft.py` 完成一次可信 SFT 复现。
- [ ] 使用官方 `grpo.py` 完成一次可信 GRPO 复现。
- [ ] B0、B1、B2 使用同一评测合同。
- [ ] 结果、日志、配置和偏差均有持久记录。
- [ ] Baseline 结论被标记为 accepted、blocked 或 waived；当前不得默认 accepted。

## 5. 进度更新规则

- 每完成一周，更新本文件的“当前阶段”和“当前周”。
- 每次路线、模型或评测合同变化，先更新子项目 `PLAN.md`，再修改本文件。
- 子项目内部的小问题不写入父目录；只有路线变化、阶段门状态和重大阻塞写入这里。
- 未经验证的分数不得写成可信 baseline 指标。
