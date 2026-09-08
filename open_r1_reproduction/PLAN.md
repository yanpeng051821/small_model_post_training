# Open-R1 Baseline Plan

最后更新：2026-07-30
状态：B0 结果证据待补；B1 训练已报告完成，配置/checkpoint/LightEval 证据待同步；B2 未运行
权威规则：模型、数据、评测和执行偏差以本文件为准；周知识文档只解释原理，不另建一套合同。

## 1. 使命与范围

目标是在单张 T4 16GB 上，使用 Open-R1 官方代码路径完成 Qwen3-0.6B 的缩放后训练闭环：

```text
B0 Base Eval
-> Reasoning SFT
-> B1 Eval
-> GRPO / RLVR
-> B2 Eval
-> 同条件归因与 verification
```

核心产出不是最高分，而是一条可重复、可解释、可比较的 baseline。该子项目使用固定版本 Open-R1/TRL 作为真实集成实现，不在这里重写 Trainer 或 rollout engine；项目级主路线会在后续课程、算法练习和 CS336 终验中检查独立实现能力。该路线不宣称完整复现 DeepSeek-R1 的大规模多阶段训练。

三者职责如下：

- 独立参考实现：证明掌握公式、shape、mask、loss 和更新语义。
- 固定版本 TRL：提供数值对照和实际执行语义。
- Open-R1：提供真实数据、配置、训练入口、评测和系统集成。

## 2. 已冻结合同

### 2.1 身份与源码

| 字段 | 固定值 |
|---|---|
| route | `reproduce` |
| baseline id | `open-r1-qwen3-0.6b` |
| variant id | `t4-scaled-sft-grpo` |
| source repo | `huggingface/open-r1` |
| source directory | `D:\pythonlearning\open-r1` |
| source commit | `1416fa0cf21595d2083b399a2a0bbddd7f6e9563` |
| Open-R1 / TRL / Transformers | `0.1.0.dev0` / `0.18.0` / `4.52.3` |
| Python route | Python 3.11 + `uv` |

### 2.2 模型链

| ID | 初始化 | 训练 | 作用 |
|---|---|---|---|
| B0 | `Qwen/Qwen3-0.6B-Base` 预训练权重 | 无 | 原始能力对照 |
| B1 | B0 权重 | 官方 `src/open_r1/sft.py` | 测量 Reasoning SFT 的变化 |
| B2 | 本地 B1 checkpoint | 官方 `src/open_r1/grpo.py` | 测量 GRPO/RLVR 的增量 |

B2 使用新的 GRPO optimizer、scheduler 和输出目录，不恢复 SFT optimizer，也不覆盖 B1。

### 2.3 数据

| 阶段 | 数据集 | config/split | 字段与规则 |
|---|---|---|---|
| SFT | `open-r1/Mixture-of-Thoughts` | `math` / `train` | 保留官方 `messages` 语义；baseline 使用 full-sequence LM loss |
| GRPO | `open-r1/OpenR1-Math-220k` | `default` / `train` | `problem` 进入 prompt，`solution` 仅供 accuracy reward |

两阶段子集均采用可重复规则：固定数据版本或 fingerprint，`shuffle(seed=42)` 后选择前 N 条。N 不凭感觉填写，由 Week 2 吞吐与显存测量决定并在首次正式运行前写回本文件。

### 2.4 Reward 与评测

- GRPO reward：`accuracy + format + tag_count`；任何权重变化必须记录。
- 主 benchmark：MATH-500 pass@1 / accuracy。
- 次 benchmark：AIME 2024，仅在预算允许时执行，不作为 baseline 最低验收条件。
- 行为指标：format rate、EOS rate、平均输出 token、截断比例。
- 效率指标：latency、throughput、peak VRAM。
- 训练诊断：loss、各 reward mean/std、zero-std group、KL、clip ratio、NaN。
- 独立评估入口：单 GPU 使用 LightEval CLI；`trainer.evaluate()` 仅作为训练内诊断，不能替代 benchmark。
- B0/B1/B2 必须使用同一 benchmark、chat template、system prompt、生成参数、响应数、seed policy、答案提取和指标聚合方式。

### 2.5 Seed policy

- 数据 shuffle seed：`42`。
- 训练 seed：`42`，并记录实际 CLI/YAML、CUDA 与库版本。
- 评估使用固定 seed 或固定 seed 列表；具体列表长度随 canonical 响应数一起在 Week 2 写回。
- 单次训练结果不冒充统计显著性；额外 seeds 属于 baseline 后的稳健性实验。

## 3. 测量后冻结的门控值

以下字段不是任意 TBD，而是已经固定了决策时点和选择规则：

| 字段 | 首选路线 | 测量与 fallback | 最晚冻结时间 |
|---|---|---|---|
| precision | FP16 | 若固定依赖/算子不兼容则记录替代方案；T4 不默认使用 BF16 | 首次 forward 前 |
| SFT 方式 | 官方全参数语义 | 若实测 OOM，切换 LoRA 并建立显式 variant，不伪装成同一 recipe | SFT smoke 后 |
| GRPO rollout | 官方 Transformers 路径优先 | vLLM/colocate 仅在兼容且显存可行时启用 | GRPO smoke 前 |
| SFT/GRPO 子集 N | 固定规则后的最大可信规模 | 依据 tokens/s、step time、峰值显存和可用预算选择 | 正式 run 前 |
| max prompt/completion length | 以数据分位数和答案/EOS 保留率决定 | 若缩短，报告截断率和语义偏差 | 各 smoke 前 |
| MATH-500 响应数 | 上游可比路线优先使用每题 4 个响应估计 pass@1 | 若 T4 预算不允许，三模型统一降为 1，并标记 scaled deviation | B0 canonical eval 前 |
| chat template / EOS / PAD | 三模型复用同一显式合同 | 先打印目标 tokenizer 和一条真实样本，不猜 token id | B0 eval 与训练前 |

## 4. 执行顺序

1. Week 2：用 `uv` 建立环境；审计 GPU、CUDA、依赖、模型/tokenizer 与一条真实数据。
2. 固定 chat template、EOS/PAD、评估生成参数，运行 B0 smoke 与 canonical eval。
3. Week 3-4：完成 SFT 数据审计、smoke 与 B1 正式缩放训练。
4. Week 5：对 B1 使用与 B0 相同的独立评测合同，并做失败分析。
5. Week 6-7：从 B1 启动新的 GRPO run，完成 smoke 与 B2 缩放训练。
6. Week 8：同条件评测 B2，比较 B0/B1/B2，完成 verification 与 baseline 判定。

默认执行原则：先一条真实样本，再最小 smoke，再正式运行；同类失败只做一次有证据的修复重试。官方逻辑只有出现明确兼容问题时才做最小修改，并记录 diff。

## 5. 路径、输出与验收

- baseline root：`baselines/local/open-r1-qwen3-0.6b/`。
- durable logs：`baselines/local/open-r1-qwen3-0.6b/logs/`。
- 必需输出：环境清单、源码身份、最终 YAML/CLI、B0/B1/B2 checkpoint 指针、训练日志、逐题评测输出、汇总指标、失败样本和 `verification.md`。
- 验收条件：三模型在同一评测合同下得到可追溯结果；无未解释 NaN；数据、配置、源码和偏差有记录；结论由日志与样本支持。
- baseline 可以得到提升、无变化或退化；只要执行与验证可信，结果都可以被接受并分类。不能保持可比性时标记 `operational_but_incomparable`，核心路线在资源内不可行时标记 `blocked`。

## 6. 风险与 fallback

- T4 与官方多 H100 recipe 在精度、显存和吞吐上不同，绝对分数不可直接宣称复现。
- 若首次 forward OOM，按顺序缩小 micro-batch、启用 gradient checkpointing、调整长度；改变数据规模或切换 LoRA 必须建立并记录偏差。
- 若 vLLM、PyTorch、CUDA 不兼容，优先保留 Transformers rollout 的正确性，再考虑性能。
- 若长样本截断严重，不能只提高 `max_length`；需同时报告长度分位数、答案/EOS 保留率与显存代价。
- 自定义教育数据、Agent、APO/OPD、assistant-only 消融和 pure RL 均在 baseline gate 之后。

## 7. Revision Log

| Time | What changed | Why | Impact |
|---|---|---|---|
| 2026-07-19 | 建立 reproduce 路线并固定 source commit | 先复用可信开源实现 | 尚未启动 setup 或训练 |
| 2026-07-23 | 冻结 B2 从 B1 初始化、SFT `math` config | 保持阶段归因并聚焦数学推理 | 明确 B0/B1/B2 模型链 |
| 2026-07-23 | 整理唯一复现合同与测量门 | 避免旧 TBD 和多文档冲突 | Week 1 关闭，进入 Week 2 setup |
| 2026-07-30 | 将本计划明确为项目级 L0 八周运行子路线 | 后续依次增加端到端系统、基础补缺、课程实战、CS336 终验和论文复现 | 不改变 B0/B1/B2 合同 |

## 8. Checklist

- living state：[`CHECKLIST.md`](CHECKLIST.md)
- Week 1 学习与源码依据：[`weeks/week_01_source_and_contract.md`](weeks/week_01_source_and_contract.md)
