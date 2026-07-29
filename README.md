# 小参数模型后训练专项

本目录用于开展 `0.6B-3B` 小参数语言模型的后训练研究与实验，核心方向是：

> 在固定数据、算力和评测预算下，比较 Reasoning SFT、偏好优化、RLVR/GRPO 与蒸馏对小模型推理、工具调用和输出效率的影响。

## 当前文档

- [`small_model_post_training_research_and_roadmap.md`](small_model_post_training_research_and_roadmap.md)：学习进展审阅、行业与招聘调研、专项研究问题和八周路线。
- [`PROGRESS.md`](PROGRESS.md)：父目录级专项进度、阶段门和当前活跃主线。
- [`open_r1_reproduction/README.md`](open_r1_reproduction/README.md)：当前活跃的 Open-R1 学习与复现子项目。
- [`experiments/README.md`](experiments/README.md)：后续实验记录的命名与最低记录要求。

## 当前主线

```text
当前阶段：可信开源 baseline 复现
当前子项目：Open-R1 + Qwen3-0.6B-Base
当前周：Week 1 - 源码地图与复现合同
当前状态：进行中，尚未执行训练
```

父目录只负责方向、阶段门和跨子项目进度；Open-R1 的源码阅读、环境、运行、指标和每周复盘全部记录在 `open_r1_reproduction/`。

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
├── open_r1_reproduction/
│   ├── README.md
│   ├── PLAN.md
│   ├── CHECKLIST.md
│   ├── open_r1_8_week_roadmap.md
│   └── weeks/
└── experiments/
    └── README.md
```

数据集、评测代码和训练代码在确定第一阶段实验方案后再创建，避免在研究问题固定前堆叠空目录。
