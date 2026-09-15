# 小参数模型后训练专项

> 本目录是六个月 LLM 算法工程师转型路线的专业主线；本目录保留算法和实验细节，半年优先级以总路线为准。

本目录用于开展 `0.6B-3B` 小参数语言模型的后训练研究与实验，核心方向是：在固定数据、算力和评测预算下，比较 Reasoning SFT、偏好优化、RLVR/GRPO 与蒸馏对小模型推理、工具调用和输出效率的影响。

## 当前文档

- [`small_model_post_training_research_and_roadmap.md`](small_model_post_training_research_and_roadmap.md)：项目级七阶段主路线；从 Open-R1 真实复现走到 CS336 终验和论文研究。
- [`PROGRESS.md`](PROGRESS.md)：父目录级专项进度、阶段门和当前活跃主线。
- [`independent_implementation/README.md`](independent_implementation/README.md)：SFT、token log-prob、DPO 和后续 GRPO 的独立实现入口。
- [`open_r1_reproduction/README.md`](open_r1_reproduction/README.md)：当前活跃的 Open-R1 学习与复现子项目。
- [`experiments/README.md`](experiments/README.md)：后续实验记录的命名与最低记录要求。

## 当前主线

```text
当前阶段：核心后训练算法独立实现
当前主任务：关闭 Gate 0B 真实 SFT 工程实验
并行子项目：Open-R1 + Qwen3-0.6B-Base 证据收尾
当前状态：SFT Gate 0A 本机核心已完成；按执行计划完成数据审计、B0、shadow、GPU smoke、正式 S1 和训练后评测，再进入 token log-prob
```

父目录的项目级主路线负责“依次学什么、学到什么程度”；Open-R1 八周运行子路线负责“服务器上如何得到可信 B0/B1/B2”。两者现在并行：独立 SFT 是主任务，Open-R1 只补证据。不能把跑通 Open-R1 等同于已经完成后训练基础阶段。

## 计划中的实验主线

暂定项目名：`TinyTutor-PostTrain`

候选模型：

- 主实验：Qwen3-0.6B 或 MiniCPM5-1B
- 扩展对照：SmolLM 系列 1.7B/3B
- 训练方式：优先 LoRA，在现有 T4 16GB 环境建立可复现闭环

计划比较：

1. 原始 Instruct/Base baseline
2. 普通 SFT 与 Reasoning SFT
3. DPO/APO
4. GRPO/RLVR
5. 离线蒸馏；条件允许时扩展 On-Policy Distillation

## 目录约定

```text
small_model_post_training/
├── README.md
├── PROGRESS.md
├── small_model_post_training_research_and_roadmap.md
├── independent_implementation/
│   ├── README.md
│   ├── EXPERIMENT_CONTRACT.md
│   └── REAL_SFT_EXECUTION_PLAN.md
├── open_r1_reproduction/
│   ├── README.md
│   ├── PLAN.md
│   ├── CHECKLIST.md
│   ├── open_r1_8_week_roadmap.md
│   └── weeks/
└── experiments/
    └── README.md
```

Gate 0B 的配置、数据审计、评测和训练模块只在执行到对应阶段时创建，避免提前堆叠空目录。
