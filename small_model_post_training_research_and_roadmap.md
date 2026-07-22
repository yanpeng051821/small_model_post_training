# 小参数模型后训练：研究与学习路线

调研日期：2026-07-19
目标范围：稠密模型约 `0.3B-8B`，或 MoE 激活参数不超过约 `4B`

## 1. 结论

当前已经具备进入“小参数模型后训练”方向的理论基础。下一阶段不应继续横向收集概念，而应完成一个可复现的 `0.6B-3B` 后训练实验项目。

建议将专项收敛为：

> `1B` 左右教育工具 Agent 的推理后训练：在固定算力和数据预算下，比较 Reasoning SFT、偏好优化、RLVR/GRPO 与蒸馏对工具成功率、推理正确率、能力遗忘和输出成本的影响。

这条路线与现有 MiniMind 学习成果高度衔接：LoRA、知识蒸馏、DPO、GRPO、reward、trajectory 和 Agentic RL 都能直接进入实验，而不是重新从零学习。

## 2. 当前学习进展审阅

### 2.1 路线图状态

MiniMind 路线图顶部总表存在滞后：

- 第 1-10 周，以及插入的第 6.5、7.5 周已经完成。
- 第 11 周总表仍写“进行中”，但详细章节已经标记为“已完成”。
- 第 11 周的 GRPO、PPO、advantage、ratio、clip、KL 等核心任务已经完成。
- 第 12 周并非“未开始”，详细状态已经是“进行中”，并已有完整的 Agentic RL 理论文档。

本地依据：

- [`../model_training_12_weeks/minimind_12_week_roadmap.md`](../model_training_12_weeks/minimind_12_week_roadmap.md)
- [`../model_training_12_weeks/week_11_grpo.md`](../model_training_12_weeks/week_11_grpo.md)
- [`../model_training_12_weeks/week_11_ppo_prerequisites.md`](../model_training_12_weeks/week_11_ppo_prerequisites.md)
- [`../model_training_12_weeks/week_12_agentic_rl.md`](../model_training_12_weeks/week_12_agentic_rl.md)

### 2.2 当前真正的缺口

第 12 周实践任务仍未形成闭环：

- 尚未准备完整的 Agent RL 数据、基础权重和 reward model。
- 尚未保存 tool-call 训练前 baseline。
- 尚未跑通最小 `train_agent.py --debug_mode`。
- 尚未保存并分析一条真实多轮 trajectory。
- 尚未验证错误工具、错误参数、unfinished 等轨迹的 reward 排序。
- 尚未完成训练前后同条件评估。

本地依据：

- [`../model_training_12_weeks/week_12_agentic_rl_practice_todo.md`](../model_training_12_weeks/week_12_agentic_rl_practice_todo.md)

因此当前能力应拆成两个维度：

| 维度 | 当前判断 |
|---|---|
| 后训练概念与源码阅读 | 已完成入门，可以进入专项研究 |
| 实验与研究工程能力 | 尚未闭环，是当前主要短板 |

下一阶段最重要的不是再证明“读懂了算法”，而是证明能够设计数据、运行训练、固定评测、分析失败并做出实验决策。

## 3. 值得关注的实验室和公司

### 3.1 OpenBMB / 面壁智能：MiniCPM

MiniCPM5-1B 面向端侧助手、代码 Agent、工具使用和推理场景，后训练采用：

```text
SFT -> 专项 RL teachers -> On-Policy Distillation
```

它是当前与本专项最贴近的案例，值得重点跟踪：

- 小模型的 Think/No-Think 双模式
- 专项 RL teacher
- On-Policy Distillation
- 多领域能力融合
- 端侧推理和工具调用评测

来源：[OpenBMB MiniCPM 官方仓库](https://github.com/OpenBMB/MiniCPM/)

### 3.2 Hugging Face：SmolLM

SmolLM3-3B 提供了较完整的开放配方：

- Reasoning mid-training
- 合成 reasoning 数据
- Reasoning/Non-reasoning 双模式 SFT
- Anchored Preference Optimization
- 模型合并与长上下文能力恢复

它最适合作为“研究实验应该如何设计、做 ablation 和记录”的参考实现。

来源：[SmolLM3 完整训练配方](https://huggingface.co/blog/smollm3)

### 3.3 阿里 Qwen

Qwen3 提供 `0.6B`、`1.7B`、`4B` 等小模型，其后训练分为四个阶段：

1. 长 CoT 冷启动
2. 推理强化学习
3. Thinking/Non-thinking 融合
4. 面向指令、格式和 Agent 能力的通用强化学习

这说明工业界后训练已经从单次 SFT/DPO 转向分阶段能力注入和融合。

来源：[Qwen3 官方技术介绍](https://qwenlm.github.io/blog/qwen3/)

### 3.4 小米 MiMo

MiMo-7B 是“小模型推理 RL”的重要参考，公开内容包括：

- RL-Zero 与 SFT 后 RL
- 数学和代码可验证奖励
- 难度感知的代码 reward
- 简单样本重采样
- 异步 rollout 与 reward 计算

它最值得学习的是算法、数据与 rollout 系统的联合设计。

来源：[XiaomiMiMo 官方仓库](https://github.com/XiaomiMiMo/MiMo)

### 3.5 Microsoft Research AI Frontiers：Phi

Phi-4 Reasoning 强调：

- teachable prompt 筛选
- 高质量教师推理轨迹
- Reasoning SFT
- outcome-based reinforcement learning
- 正确率与推理 token 成本之间的权衡

来源：[Phi-4 Reasoning Technical Report](https://www.microsoft.com/en-us/research/publication/phi-4-reasoning-technical-report/)

### 3.6 NVIDIA Nemotron

NVIDIA 的重点包括：

- Cascade RL
- 多领域 On-Policy Distillation
- Agentic RL
- rollout 生成加速
- 合成后训练数据

Nemotron-Cascade 2 总参数约 30B、激活约 3B，是“高智能密度 MoE”的重要参考。

来源：[Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)

### 3.7 Apple Foundation Models

Apple 的约 3B 端侧模型涉及：

- 蒸馏训练
- Quantization-Aware Training
- 权重、embedding 与 KV cache 低比特化
- 通过低秩适配恢复压缩损失
- 面向私有、本地和 OS 集成场景的后训练

来源：[Apple Foundation Models 2025 Updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

### 3.8 Google Gemma

Gemma 3 270M 面向超低成本定制和端侧部署，适合研究极小模型的领域适配、结构化输出、分类与蒸馏。

来源：[Gemma 3 270M](https://developers.googleblog.com/introducing-gemma-3-270m/)

### 3.9 字节跳动 Seed

Seed 的招聘和研究方向更偏向：

- 大模型强化学习系统
- 动态负载与异构资源
- 复杂 Agent/环境交互
- 多模态 RL
- 训推系统联合优化

来源：[Seed 大模型人才招聘](https://seed.bytedance.com/zh/seedearlycareer)

## 4. 当前招聘方向

企业通常不会统一使用“小模型后训练工程师”这一名称，相关岗位主要分为以下类别。

### 4.1 后训练算法 / Research Engineer

常见关键词：

- SFT、DPO/APO、RLHF、RLVR、GRPO
- Reward Model、Preference Optimization
- Reasoning、Instruction Following、Tool Use
- 数据混合、训练 recipe、ablation

Apple 的后训练岗位直接覆盖 SFT、RL、指令遵循、工具使用、推理、合成数据、reward hacking 与延迟奖励。

来源：[Apple Foundation Model Post-Training 岗位](https://jobs.apple.com/en-us/details/200645804/aiml-machine-learning-researcher-post-training-for-foundation-models)

### 4.2 后训练数据 / 合成数据

常见关键词：

- Teacher generation
- 数据清洗、去重和难度分层
- Curriculum learning
- Reasoning trace、preference pair
- 自动质量评估
- 数据污染和 benchmark leakage

NVIDIA 的相关岗位明确要求构建服务 Nemotron 预训练与后训练的合成数据流水线。

来源：[NVIDIA Synthetic Data Generation 岗位](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Scientist--Synthetic-Data-Generation_JR2019460)

### 4.3 RL Training Infra / Rollout 系统

常见关键词：

- vLLM、SGLang、Ray
- 异步 rollout
- 训推切换和权重同步
- FSDP、DeepSpeed、ZeRO
- Packing、显存、吞吐量和稳定性
- Agent 环境并发

阿里 ROCK 已明确面向后训练、评测和数据生产构建大规模 Agent 训练环境。

来源：[阿里 ROCK 与 AI Infra 招聘](https://alibaba.github.io/ROCK/zh-Hans/careers/)

### 4.4 Evals、Reward 与 Agent Environment

常见关键词：

- Verifiable reward
- Grader、rubric、judge model
- Reward hacking
- Long-horizon evaluation
- Agent trajectory
- Tool-use success rate
- 交互环境和任务设计

来源：[OpenAI Frontier Evals & Environments 岗位](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/)

### 4.5 端侧模型压缩与适配

常见关键词：

- Distillation、QAT、PTQ
- LoRA/Adapter
- KV-cache quantization
- Core ML、ExecuTorch、ONNX、llama.cpp
- 质量、延迟、功耗和内存的 Pareto frontier

该方向与 Apple、Google、Microsoft Phi、MiniCPM 和小米的产品路径高度相关。

## 5. 推荐专项：TinyTutor-PostTrain

### 5.1 研究问题

> 在相同 GPU 小时和相同数据预算下，Reasoning SFT、DPO/APO、GRPO/RLVR、On-Policy Distillation 中，哪一种最能提高 `0.6B-1B` 模型的教育工具调用成功率，同时减少能力遗忘和无效长推理？

### 5.2 候选模型

- 主实验：Qwen3-0.6B 或 MiniCPM5-1B
- 扩展对照：SmolLM 系列 1.7B/3B
- 教师模型：较强的开放推理模型，或许可允许用于数据生成的 API
- 训练方式：现有 T4 16GB 环境优先 LoRA

不建议第一阶段直接从 7B full-parameter RL 开始，因为它会把主要时间消耗在显存和系统问题上，削弱对后训练机制的研究。

### 5.3 实验变量

| 实验 | 训练方式 | 主要目的 |
|---|---|---|
| B0 | 原始 Base/Instruct | 建立能力、成本和失败 baseline |
| B1 | 普通 SFT | 验证领域数据带来的基础收益 |
| B2 | Reasoning SFT | 验证推理轨迹的收益与输出成本 |
| B3 | DPO/APO | 修复格式、偏好、冗长与错误工具选择 |
| B4 | GRPO/RLVR | 使用可验证奖励优化任务成功率 |
| B5 | Distillation/OPD | 比较蒸馏与直接 RL 的样本效率 |

### 5.4 评测合同

主要指标：

- 端到端任务成功率
- 工具选择正确率
- 工具参数完全匹配率
- 最终答案 exact match / 可验证正确率

次要指标：

- 非法 tool call 率
- unfinished / 循环调用率
- 平均输出 token
- P50/P95 延迟
- 峰值显存与训练吞吐量
- 通用能力遗忘
- Reward 与实际任务成功率的相关性

公平比较要求：

- 固定初始模型
- 固定 seed prompts 与评测集
- 固定训练 token 或 GPU-hour 预算
- 固定生成参数
- 分离训练、验证和测试任务
- 保存随机种子、代码版本、数据版本与完整配置

## 6. 八周执行路线

### 第 1 周：补完现有 Week 12 实践

- 跑通 `eval_toolcall.py` baseline。
- 跑通最小 GRPO/Agent RL smoke test。
- 保存一条真实多轮 trajectory。
- 解释 reward、advantage、KL、action mask 和 observation mask。
- 验证正确轨迹、错误工具、错误参数、unfinished 的 reward 排序。

### 第 2 周：建立教育工具评测集

设计约 100-200 个可验证任务，覆盖：

- 工具选择
- 参数抽取
- 多轮 observation 利用
- 最终答案正确性
- 错误工具和错误参数
- 循环调用
- 不应调用工具时的克制能力

### 第 3 周：构造后训练数据

从同一批 seed prompts 生成：

- 普通 SFT 数据
- Reasoning SFT 数据
- Reasoning/Non-reasoning 混合数据
- chosen/rejected 偏好对
- 带可验证答案的 RL prompts

同时记录教师模型、生成参数、过滤规则和拒绝样本原因。

### 第 4 周：SFT baseline

比较普通 SFT 与 Reasoning SFT：

- 正确率
- 工具成功率
- 平均输出 token
- 推理延迟
- 格式错误
- 失败类型

### 第 5 周：DPO/APO

重点回答：

- 能否降低非法格式和错误工具选择？
- 能否控制无效长推理？
- 是否出现基础能力损伤？
- 对偏好数据噪声是否敏感？

### 第 6 周：GRPO/RLVR

使用规则型可验证 reward，分析：

- Reward 各分项
- 同 prompt 的组内 reward 方差
- Advantage 分布
- KL 与策略漂移
- 输出长度变化
- Reward hacking 案例

### 第 7 周：蒸馏对照

- 先完成离线 reasoning distillation。
- 比较 teacher 数据质量、温度和数据量。
- 条件允许时实现简化 On-Policy Distillation。
- 比较蒸馏与 RLVR 的样本效率和最终能力。

### 第 8 周：形成公开作品

最终成果应包括：

- 数据集与 dataset card
- 可重复训练配置
- Eval harness
- 多组 adapter/checkpoint
- 训练曲线和系统指标
- 失败案例分类
- 技术报告
- 一键复现实验说明

## 7. 项目验收标准

专项不能只证明“成功微调了一个聊天模型”，而应证明：

1. 提出了边界清楚、可以证伪的研究问题。
2. 建立了训练前 baseline。
3. 固定了公平的评测合同。
4. 比较了至少三种后训练方法。
5. 记录了质量、成本和系统指标。
6. 分析了失败机制，而不只报告平均分。
7. 所有结论均能由配置、日志和样本复现。
8. 能说明下一轮实验为何值得进行。

达到这些标准后，这个专项就不仅是学习练习，也可以作为后训练算法、数据、评测或 RL Infra 岗位的作品集项目。
