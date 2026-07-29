# Open-R1 缩放复现

本子项目用于在单张 T4 16GB 上建立一条可解释、可复现的小模型后训练基线：理解源码与算法，但复用 Open-R1/TRL 官方实现完成训练，不先重造后训练框架。

## 1. 项目使命

```text
Qwen3-0.6B-Base
-> Reasoning SFT
-> GRPO / RLVR
-> 独立 benchmark
-> B0 / B1 / B2 同条件归因
```

完成后应当能够从原始数据追踪到 scalar loss，运行并诊断 SFT/GRPO，使用统一评测合同解释 `B1-B0` 和 `B2-B1`，再在可信 baseline 上开展自己的小模型后训练实验。

## 2. 复现对象

- 上游仓库：[huggingface/open-r1](https://github.com/huggingface/open-r1)
- 基础模型：`Qwen/Qwen3-0.6B-Base`
- 复现路线：`Base Eval -> Reasoning SFT -> SFT Eval -> GRPO/RLVR -> Final Eval`
- 复现性质：单张 T4 16GB 条件下的缩放复现，不宣称匹配 7B/H100 官方绝对指标
- 当前状态：Week 1 已完成，进入 Week 2 setup；尚未运行训练

## 3. 范围边界

本项目要完成：

- 深入理解数据、reward、advantage、policy loss 和评估数据流。
- 复用官方 `sft.py`、`grpo.py` 与 TRL Trainer，跑通 B0/B1/B2。
- 保存配置、日志、checkpoint、评测结果与失败分析。
- 在 baseline 可信后再做 assistant-only、reward、pure RL 等消融。

本项目不做：

- 从随机参数重新预训练 0.6B 模型。
- 在 baseline 前从零重写 TRL、GRPO loss 或 rollout engine。
- 声称完整复现 DeepSeek-R1 的大规模多阶段工业流程。
- 用 TinyTutor、Agent 或自定义 reward 污染第一条 Open-R1 baseline。

## 4. 为什么先跟 Open-R1

Open-R1 已经提供 SFT、GRPO、数据生成、评测和配置入口。当前原则是：

- 不重写 Trainer。
- 不重写 GRPO loss。
- 不自建 rollout engine。
- 不先设计教育场景 reward。
- 优先通过配置缩放官方路径。
- 只有出现明确兼容性问题时才做最小修复，并记录偏差。

## 5. 文档导航

- [`PLAN.md`](PLAN.md)：唯一权威复现合同；模型、数据、评测、门控值和验收条件均以此为准。
- [`CHECKLIST.md`](CHECKLIST.md)：baseline gate 的 living checklist。
- [`open_r1_8_week_roadmap.md`](open_r1_8_week_roadmap.md)：八周总看板。
- [`weeks/week_01_source_and_contract.md`](weeks/week_01_source_and_contract.md)：Week 1 唯一知识文档，内部按知识、源码、数据流、对照、合同和进度分节。

## 6. 三个模型状态

| ID | 模型 | 说明 | 当前状态 |
|---|---|---|---|
| B0 | Qwen3-0.6B-Base | 训练前对照 | 未评测 |
| B1 | Open-R1-SFT-0.6B | Reasoning SFT 后模型 | 未训练 |
| B2 | Open-R1-GRPO-0.6B | GRPO/RLVR 后模型 | 未训练 |

## 7. 完成标准

这一轮的目标不是追求最高分，而是建立可信链路：

```text
源码与配置可追溯
-> 环境可重复
-> smoke test 通过
-> 正式缩放运行完成
-> 同条件评测
-> 偏差和失败可解释
```

完成 B0/B1/B2 验证前，不把自定义教育数据、Agent tool-use 或 APO/OPD 混入本 baseline。

Baseline 是否成功不以“分数必须上涨”为条件，而以流程可信为条件：三者使用同一评测合同，运行和结果可追溯，任何提升、退化或失败都能由日志与样本支持。
