# 小参数模型后训练专项：总进度

最后更新：2026-09-12

## 1. 当前状态

| 项目 | 当前值 |
|---|---|
| 专项目标 | 能从论文公式独立实现核心后训练算法，并用真实实验与评测验证 |
| 当前主线 | 独立 SFT 本机链路已关闭；当前执行 Gate 0B 真实 SFT 工程实验，关闭后再进入 token log-prob |
| 当前理解路线 | 完整工程地图和真实样本数据流走读已验收；当前进入数据模块的自研实现、锁定版本 TRL/Open-R1 映射和输出对照；框架对照尚未验收 |
| 基础模型 | Qwen3-0.6B-Base |
| 主路线进度 | Gate 0A 与 Gate 0B 本地准入已关闭；全量数据审计、人工复核（50/50）与 16K 长度筛选 artifact 已冻结。服务器侧已完成 preflight、最长样本显存探针、1/5/20-step smoke、阶段 D 首 batch shadow、完整 B0，以及 TRL 100-step checkpoint-50→100 恢复 pilot。S1 归属固定 TRL 栈，自研 runner 为 S1-ind 同条件对照。正式 S1 仍为 `NO-GO`：先关闭 479-step 合同、allocator telemetry GPU 预飞行与 MATH-500 评测接口预飞行。 |
| 运行子路线进度 | B1 训练已报告完成；配置、checkpoint、LightEval 和 B0/B1 对照证据待同步；B2 未运行 |
| Baseline 信任状态 | B0 已按冻结合同实测并归档：MATH-500 `pass@1:1 = 43.2%`、`pass@1:4 = 44.9%`，GSM8K `qem 47.6%`，MMLU `54.5%`，ARC-Challenge `acc_norm 45.4%`，HellaSwag `acc_norm 53.3%`，全量 validation NLL `0.7805`（1,968 条）。历史 `26-28%` 的记录不可复现，已被实测值取代 |
| 当前是否允许自定义实验 | 允许启动独立实现、tiny 合同测试和预注册受控实验；Open-R1 证据收尾不再阻塞主线，但未闭环的 B0/B1 不得写成可信对照结论 |
| B0 评测合同 | MATH-500：Qwen3 chat template、temperature `0.6`、top-p `0.95`、max_new_tokens/max_model_length `32768`、1n+4n pass@1；GSM8K 与 regression 使用各自冻结的 greedy/log-likelihood 口径 |

Open-R1 Week 1 已完成。B0 canonical eval 与 B1 训练均已由运行侧报告完成，但本地仓库仍缺少可核对的完整命令、结果摘要、checkpoint 指针和 B1 LightEval 结论；历史 4-step smoke 的 `loss=0` / `grad_norm=NaN` 与后续正式 B1 训练之间的关系也尚未归档。因此 Open-R1 继续作为并行证据收尾，不宣称 baseline 已关闭。

2026-08-23 根据近三周 Coding 诊断再次校准：不再等待 Open-R1 全部证据关闭才开始算法实现。当前主任务是独立 SFT/token log-prob，Open-R1 只并行补真实框架证据；nanochat、Happy-LLM、课程与 llm-algo-leetcode 均按后续真实缺口调用，不同时启动。

2026-08-27 Gate 0A 本机核心计算关闭：15 条 CPU 测试覆盖 causal shift、mask、不同有效长度的 token 级归约、dtype/shape 边界、PyTorch 参考 loss/梯度、SGD 单步更新、tiny overfit 和自回归复现。Gate 0B 仍等待服务器数据审计、fingerprint、B0 与环境归档，不启动正式 S1。

2026-08-29 单设备 FP32 optimizer step 核心关闭：22 条测试进一步覆盖跨 micro-batch 的全局 token 归一化、完整 batch 参数对齐、空输入、冻结参数和 gradient clipping。裁剪前 grad norm 返回、日志与硬停止策略归入后续可观测性阶段。

2026-09-01 SFT 数据与 tokenizer 合同关闭：37 条默认单元测试覆盖 right-padding、attention mask、assistant-only labels、单轮模板前缀与结束 token 边界；1 条独立集成测试验证锁定的 `Qwen3-0.6B-Base@311c62e...` 真实模板。当前不立即切换算法，先补齐 `messages -> batch -> HF causal LM -> loss -> optimizer step` 的端到端接线与最小 smoke，再进入 token log-prob。

2026-09-02 SFT 本机纵向链路关闭：结构化 batch 经 HF 风格 `model(input_ids=..., attention_mask=...) -> outputs.logits` 进入独立 loss 与 optimizer step；gradient accumulation、冻结参数、gradient clipping 旧测试已迁移到统一 batch 合同，40 条默认测试与 1 条 tokenizer 集成测试通过。真实 Qwen 权重训练、TRL shadow 对照和 Gate 0B 证据仍待服务器关闭。

2026-09-02 主线顺序调整：不在本机 SFT 核心完成后立即进入 token log-prob，先按 `independent_implementation/REAL_SFT_EXECUTION_PLAN.md` 完成真实 Qwen SFT 工程。顺序固定为本机工程骨架、数据审计、B0、首 batch shadow、20-step smoke、100-step 恢复 pilot、正式 S1、同条件训练后评测；Gate 0B 关闭后再切换算法。

2026-09-03 Gate 0B 本地准入关闭：配置、数据物化、训练器、全局 token 归一化、可观测性、失败快照、原子 checkpoint、保留上限和跨进程恢复均已接通；与 TRL 0.18 collator 及 Transformers causal loss/update 完成对齐。两次真实 OpenR1-Math 小样本审计得到相同 train/validation/rejected hash；真实 Qwen3-0.6B-Base 在 8 GB RTX 4060 Ti 上完成 BF16 两步训练与恢复，峰值约 6.53 GB。该证据不替代全量数据冻结、B0 或服务器 smoke。

2026-09-11 服务器 Gate 0B 首次会话完成：A100-80GB 上 22,295-token 最长样本训练探针失败，据此冻结 16,384-token 确定性长度筛选 artifact（train 62,208→61,224、validation 2,000→1,968，不截断，超长样本单独隔离为 `length_filtered_*.jsonl`）；服务器 preflight 12 项全部通过。完成 1-step、5-step 和 20-step smoke：20 步训练 loss 0.782→0.646、grad norm 4.58→0.70、显存峰值 39.8 GB，validation NLL 相对 base 下降 16.1%（0.6754 vs 0.8050）。生成门、flash-attn/vLLM 环境与首 batch shadow 的诊断和修复均已归档；完整 B0 五任务也已完成。

2026-09-12 TRL 100-step 恢复 pilot 已从 checkpoint-50 完成至 global step 100，exit code `0`；恢复段无 OOM、NaN 或 Inf，外部 GPU 观测峰值约 51,471 MiB。固定全量 validation NLL 从 B0 的 `0.7804832` 降至 `0.5952246`，使用同一 1,968 条 validation artifact 和 11,367,594 个有效 token。GSM8K 50-sample pilot `qem=0.52`、regression smoke 完成；这些不是正式 B0/S1 差值。MATH-500 pilot 未形成终态失败记录，评测接口需在正式 S1 前预飞行。数据盘文件 SHA-256 复核与所有剩余 S1 门禁见 [`S1_PRELAUNCH_PLAN.md`](S1_PRELAUNCH_PLAN.md)。

同日完成 B0 的通用能力回归面板与 GSM8K：MMLU 57 学科平均 `54.52%`、ARC-Challenge `acc_norm 45.39%`、HellaSwag `acc_norm 53.35%`、GSM8K `qem 47.61%`（1319 题全量），量级与 Qwen3-0.6B-Base 公开水平一致，支持评测栈配置正确。为跑通评测栈，修复了 flash-attn 安装后 vLLM 引擎无法启动的两个故障（`VLLM_WORKER_MULTIPROC_METHOD=spawn` 与 `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`），配方与陷阱已写入 `SERVER_RUNBOOK.md` §11；评测目录已区分正式证据与探针，见 `runs/gate0b-16k/b0/evals/PROBE_DIRECTORIES.md`。同日修订实验合同 §6：**S1 改为固定版本 Open-R1/TRL 训练栈**，新增 **S1-ind** 作为自研 runner 的同条件对照，解决执行计划 §6.1 记录的合同冲突。

## 2. 大阶段看板

| 阶段 | 内容 | 状态 | 完成条件 |
|---|---|---|---|
| P0 | 行业、实验室和招聘方向调研 | 已完成 | 研究方向与岗位能力地图已形成 |
| P1 | 选择可信开源基线 | 已完成 | 选定 Open-R1 + Qwen3-0.6B-Base |
| P2 | Open-R1 真实复现 | 并行收尾 | B0/B1/B2 可复现、同条件评测且证据闭环 |
| P3 | 核心算法独立实现 | 进行中 | SFT、token log-prob、DPO、GRPO 的最小实现、测试和数值对齐 |
| P4 | 端到端系统与定向补缺 | 未开始 | nanochat 完整调用链；Happy-LLM、课程和 llm-algo 只补诊断未通过项 |
| P5 | CS336 基础终验 | 未开始 | 不看答案完成 Assignment 5，DPO/GRPO 测试通过 |
| P6 | 论文复现与自主实验 | 未开始 | 根据 baseline 选题，完成复现、失败分析和自主消融 |
| P7 | TinyTutor 教育场景迁移 | 未开始 | 教育数据、reward 与 eval contract 固定 |
| P8 | 作品集与技术报告 | 未开始 | 代码、数据、指标、失败分析可复现 |

## 3. 当前活跃子项目

- 子项目入口：[`open_r1_reproduction/README.md`](open_r1_reproduction/README.md)
- 当前主任务执行计划：[`independent_implementation/REAL_SFT_EXECUTION_PLAN.md`](independent_implementation/REAL_SFT_EXECUTION_PLAN.md)
- 项目级七阶段主路线：[`small_model_post_training_research_and_roadmap.md`](small_model_post_training_research_and_roadmap.md)
- Open-R1 八周运行子路线：[`open_r1_reproduction/open_r1_8_week_roadmap.md`](open_r1_reproduction/open_r1_8_week_roadmap.md)
- Baseline 计划：[`open_r1_reproduction/PLAN.md`](open_r1_reproduction/PLAN.md)
- Baseline 检查清单：[`open_r1_reproduction/CHECKLIST.md`](open_r1_reproduction/CHECKLIST.md)

## 4. 阶段门

进入 TinyTutor 自定义实验前，学习与运行两条线必须同时满足。当前先关闭 SFT 门，不要求下列所有事项同时展开：

- [x] 固定上游源码 commit 和环境版本。
- [x] 为 Qwen3-0.6B-Base 训练前 baseline 补齐可核对的结果摘要与证据指针。（已按冻结合同实测：MATH-500 `pass@1:1 43.2%`、GSM8K `47.6%`、MMLU `54.5%`、ARC `45.4%`、HellaSwag `53.3%`；证据 `evidence/gate0b-16k-b0-*.json`）
- [ ] 独立实现单卡 SFT，并与固定版本 TRL 完成数值对齐。
- [ ] 独立实现 DPO 核心 loss 与数据流，并通过不变量测试。
- [ ] 独立实现 GRPO rollout、advantage、KL/clip/loss 和更新闭环，并完成数值对齐。
- [ ] 使用官方 `sft.py` 完成一次可信 SFT 复现。
- [ ] 使用官方 `grpo.py` 完成一次可信 GRPO 复现。
- [ ] B0、B1、B2 使用同一评测合同。
- [ ] 完成 nanochat 端到端追踪和一次有验证的修改。
- [ ] 完成 Happy-LLM 基础诊断，只补未通过项。
- [ ] 完成 DeepLearning.AI 与北大课程的核心后训练实践。
- [ ] 完成 llm-algo-leetcode 中训练、对齐、反向传播、显存和分布式相关练习。
- [ ] 不看答案完成 CS336 Assignment 5，要求测试全部通过。
- [ ] 完成至少一次论文级受控复现和一个自主消融。
- [ ] 结果、日志、配置和偏差均有持久记录。
- [ ] Baseline 结论被标记为 accepted、blocked 或 waived；当前不得默认 accepted。

## 5. 进度更新规则

- 每通过一个里程碑，分别更新本文件的“主路线进度”和“运行子路线进度”。
- 每次路线、模型或评测合同变化，先更新子项目 `PLAN.md`，再修改本文件。
- 子项目内部的小问题不写入父目录；只有路线变化、阶段门状态和重大阻塞写入这里。
- 未经验证的分数不得写成可信 baseline 指标。
