# 小参数模型后训练：研究与学习路线

调研日期：2026-07-19
路线修订：2026-08-23
目标范围：稠密模型约 `0.3B-8B`，或 MoE 激活参数不超过约 `4B`

## 1. 结论

当前已经具备进入“小参数模型后训练”方向的理论基础。下一阶段不能只停留在调用训练框架，也不应继续横向收集概念，而应同时建立两种能力：

1. 不依赖 Trainer 封装，独立实现单卡参考版 SFT、DPO、PPO 核心损失和 GRPO 训练闭环。
2. 使用 Open-R1/TRL 完成真实模型实验，并通过固定评测合同形成可信证据。

建议将专项收敛为：

> `1B` 左右教育工具 Agent 的推理后训练：在固定算力和数据预算下，比较 Reasoning SFT、偏好优化、RLVR/GRPO 与蒸馏对工具成功率、推理正确率、能力遗忘和输出成本的影响。

这条路线与现有 MiniMind 学习成果高度衔接：LoRA、知识蒸馏、DPO、GRPO、reward、trajectory 和 Agentic RL 都能直接进入实验，而不是重新从零学习。

最终目标不是“会配置 TRL”，而是能够从论文公式或伪代码出发，写出最小正确实现，用单元测试和数值对齐证明实现可信，再接入成熟框架完成规模化实验、复现论文并提出自己的消融假设。

## 2. 当前学习进展审阅

### 2.1 路线图状态

MiniMind 路线图状态已在 2026-08-23 对齐：

- 第 1-11 周，以及插入的第 6.5、7.5 周已经完成。
- 第 11 周的 GRPO、PPO、advantage、ratio、clip、KL 等核心任务已经完成。
- 第 12 周理论主线已完成；真实 `train_agent.py`、轨迹调试、`eval_toolcall.py` 和 reward 观察保留在独立实践待办中。
- MiniMind 十二周用于证明概念与数据流理解，不替代当前 SFT/DPO/GRPO 的独立实现验收。

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

## 6. 项目级学习路线

本路线不再按固定周数推进，而使用带验收门的阶段。一次只激活一个阶段；已经掌握的内容通过诊断后跳过，不能因为“看完了资料”就标记完成。

### 6.1 路线依据与分工

| 资源 | 在本项目中的职责 | 使用方式 |
|---|---|---|
| [Open-R1](https://github.com/huggingface/open-r1) | 真实后训练、评测与工程集成基线 | 完整完成当前 B0/B1/B2 子路线 |
| [Karpathy nanochat](https://github.com/karpathy/nanochat) | tokenizer、预训练、微调、评测、推理的最小端到端系统 | 追踪并修改完整调用链，不以高成本复训 GPT-2 为目标 |
| [Happy-LLM](https://github.com/datawhalechina/happy-llm) | Transformer、LLaMA、Tokenizer、预训练和 SFT 基础查漏 | 先诊断，已掌握部分直接跳过 |
| [DeepLearning.AI 后训练课程](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) | 数据、方法选择、eval、错误分析与生产闭环 | 形成方法选择和评测检查表 |
| [北大后训练实践课程](https://posttrain.gaozhijun.me/docs/) | SFT、DPO、GRPO、蒸馏与综合项目的课程化实践 | 补齐统一模型下的方法对照和实验报告 |
| [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) | PyTorch 算法、测试、反向传播、显存与分布式练习 | 选择训练与对齐相关题目闭卷完成 |
| [Stanford CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment) | DPO/GRPO 独立实现终验 | 不提前看答案，直到测试通过 |

3Blue1Brown、Karpathy Zero to Hero 等基础材料不单列为必修阶段。只有诊断发现梯度、交叉熵、反向传播或注意力直觉存在缺口时，才定向回补。

### 6.2 七阶段看板

| 阶段 | 主题 | 当前状态 | 核心产出与通过条件 |
|---|---|---|---|
| L0 | Open-R1 真实复现 | 并行收尾 | B0/B1/B2 使用同一评测合同；配置、日志、checkpoint、失败样本和 verdict 可追溯 |
| L1 | 独立 SFT 与 token log-prob | 进行中 | 不依赖 Trainer 完成 shift、mask、loss、单卡更新、极小数据过拟合和数值对齐 |
| L2 | 独立 DPO | 未开始 | 从 preference batch 到 chosen/rejected sequence log-prob 和 DPO loss；通过不变量与参考实现对齐 |
| L3 | 独立 GRPO | 未开始 | 完成 rollout、reward、group advantage、policy/reference log-prob、KL/clip/loss 和更新闭环 |
| L4 | 端到端系统与定向补缺 | 未开始 | 追踪并修改 nanochat；Happy-LLM、后训练课程和 llm-algo 只补当前实现暴露的缺口 |
| L5 | CS336 Assignment 5 终验 | 未开始 | 独立补全 DPO/GRPO；全部要求测试通过；能把论文公式、shape、mask 和实现逐项对应 |
| L6 | 论文复现与自主实验 | 未开始 | 根据可信 baseline 的失败现象选论文，完成预注册、复现、消融、失败分析和一个自主假设 |

Open-R1 的八周文件仍是 L0 的服务器运行子路线：[`open_r1_reproduction/open_r1_8_week_roadmap.md`](open_r1_reproduction/open_r1_8_week_roadmap.md)。L0 与 L1 可以并行，但不能同时启动 nanochat、Happy-LLM、完整课程和其他新路线；Open-R1 负责框架实证，L1 负责算法独立实现。

### 6.3 学习与验收方法

每个核心方法都经过四遍：

1. **公式与数据流**：从原始样本走到 scalar loss，标出 shape、mask 和语义。
2. **闭卷最小实现**：先不照抄 Trainer 或参考答案，独立写出 PyTorch 参考版。
3. **测试与数值对齐**：覆盖边界条件，并与手算结果或固定版本实现对齐。
4. **真实实验与评测**：运行模型，分析质量、稳定性、资源消耗和失败样本。

Eval 不是最终阶段才做的任务。每一次 SFT、DPO 和 GRPO 实验都必须先固定训练前 baseline、数据切分和评测合同，训练后再进行同条件比较。

核心算法由学习者先完成首版；AI 负责调用链讲解、测试合同、代码审查、边界条件和数值验证。基础设施可以协助搭建，但不能在首次尝试前直接代写全部核心 loss 和 trainer。

### 6.4 基础阶段毕业门

L5 关闭后，才判定具备进入论文复现阶段的基础。最低能力包括：

- 从一条原始样本追踪到 scalar loss、梯度和参数更新。
- 独立实现并测试 SFT、DPO 和 GRPO 的核心数据流与 loss。
- 解释 PPO 中 old/current/reference policy、critic、GAE、ratio 和 clip。
- 诊断 AMP、梯度累积、NaN、OOM、checkpoint/resume 和常见数据错误。
- 建立训练前 baseline，并保证训练前后评测合同一致。
- 阅读论文公式，定位作者代码中的实现，并设计对照实验和消融。

这表示可以在单卡或小规模算力下开始论文复现，不表示已经掌握大规模分布式 RL、异步 rollout、CUDA/Triton kernel 或工业级数据生产；这些应在真实研究需要出现后再作为专项补充。

### 6.5 论文研究循环

进入 L6 后，不再无限增加“知识周”，而反复执行：

1. 根据 baseline 失败现象选择论文，拆公式并预注册假设、指标、预算和停止条件。
2. 独立实现论文核心增量，先复现作者 baseline，再做受控主实验。
3. 做消融、鲁棒性和失败分析，形成自己的解释或下一轮改进方向。

Dr. GRPO、DAPO 等只保留为候选，不预先指定首篇论文。最终选择必须由 L0-L5 暴露的问题、可验证性和实际算力共同决定。

## 7. 项目验收标准

专项不能只证明“成功微调了一个聊天模型”或“会调用 Trainer”，而应证明：

1. 提出了边界清楚、可以证伪的研究问题。
2. 建立了训练前 baseline。
3. 固定了公平的评测合同。
4. 关闭 L0-L4，并留下可核对的代码、测试、实验和查漏记录。
5. 不依赖参考答案完成 CS336 Assignment 5，作为基础阶段终验。
6. 能独立写出 SFT、DPO 和 GRPO 的单卡核心实现，并解释 PPO 的完整更新语义。
7. 核心实现通过边界测试、手算样例和固定版本框架的数值对齐。
8. 至少完成一次论文级受控复现，而不只是复跑作者命令。
9. 记录质量、成本和系统指标，并分析失败机制，而不只报告平均分。
10. 所有结论均能由代码、配置、日志和样本复现，并能提出自己的消融假设。

达到这些标准后，这个专项就不仅是学习练习，也可以作为后训练算法、数据、评测或 RL Infra 岗位的作品集项目。
