# Week 1：后训练知识、源码地图与复现合同

状态：已完成
开始日期：2026-07-19
最后学习同步：2026-07-23
本周原则：先理解知识和数据流，再安装环境和运行训练。

## 1. 本周学习目标

完成本周后，应当能够：

- 区分 DeepSeek-R1、Open-R1 与我们的缩放实验。
- 解释 B0、B1、B2 的作用，以及为什么先 SFT 再 GRPO。
- 解释 SFT 中 chat template、input_ids、labels 和 loss 的关系。
- 解释 GRPO 中 completion、reward、advantage、reference policy 和 policy loss 的关系。
- 找到 Open-R1 的 SFT、GRPO、reward、数据和评估入口。
- 说明 MiniMind 与 Open-R1 的关键实现差异。
- 明确哪些实验设置已经冻结，哪些必须经过 Week 2 实测后再冻结。

本周不安装依赖、不下载模型、不启动正式训练。

---

## 2. 第一节：我们在复现什么

### 2.1 三个容易混淆的对象

- **DeepSeek-R1**：研究工作和方法来源，包含 pure RL、cold-start、多阶段训练与小模型蒸馏。
- **Open-R1**：Hugging Face 对公开 R1 路线的工程化复现，提供 SFT、GRPO、reward、数据与评估代码。
- **我们的实验**：在单张 T4 16GB 上，用 Open-R1 官方代码路径训练 Qwen3-0.6B。

### 2.2 三个模型状态

```text
B0：Qwen3-0.6B-Base
        ↓ Reasoning SFT
B1：Open-R1-SFT-0.6B
        ↓ GRPO / RLVR
B2：Open-R1-GRPO-0.6B
```

- B0 是预训练后的 base model。它已有语言建模、语义理解和续写能力，但不一定能稳定遵循指令或输出规定的推理格式。
- B1 是 Reasoning SFT 模型。它不只是学“像助手一样对话”，还学习指令跟随、推理轨迹以及 `<think>/<answer>` 等回答结构。
- B2 是 GRPO / RLVR 模型。它在 B1 的起点上，提高可验证正确、格式合规回答的相对概率，而不是笼统地“让回答更好”。

在实验归因上：

- B0 回答“原始 base model 本来会多少”。
- B1 回答“SFT 具体教会了模型什么”。
- B2 回答“在 B1 基础上，GRPO 又改变了什么”。

没有 B0/B1 的单独评估，就无法把 B2 的变化归因给 GRPO。

### 2.3 为什么暂时不从 Base 直接做 pure GRPO

SFT 并不是理论上不可省略：R1-Zero 已经说明可以从 Base 直接做 pure RL。我们的选择是工程和算力条件下的 baseline 决策，而不是把“SFT 必须存在”写成普遍规律。

从 Base 直接做 RL 更接近 R1-Zero，但对单张 T4 上的 0.6B 小模型风险更高：

- 初始回答可能几乎全部错误；
- 同组 reward 可能全部相同，缺少相对学习信号；
- 输出格式不稳定；
- rollout 成本较高；
- 很难区分算法失败和模型起点太弱。

所以当前选择：先用 SFT 提高初始成功率、格式稳定性和组内 reward 差异，再用 GRPO 优化可验证能力。这能让有限 rollout 预算产生更有效的学习信号。

> 核心理解：SFT 教模型“好推理大致长什么样”，GRPO 增加高 reward 回答出现的概率。

---

## 3. 第二节：SFT 的核心知识

### 3.1 SFT 仍然是 next-token prediction

因果语言模型训练目标可以简化为：

```text
L_SFT = - sum_t m_t * log p_theta(x_t | x_<t)
```

其中 `m_t` 是 loss mask：

- `m_t = 1`：token 参与 loss，会产生梯度。
- `m_t = 0`：label 通常为 `-100`，不参与 loss。

因此，理解 SFT 不能只看原始 JSON，还要看模板、tokenization 和 labels。

### 3.2 Chat template 为什么重要

Chat template 决定：

- system/user/assistant 的边界；
- role token 和分隔符；
- `<think>`、`<answer>` 如何出现；
- generation prompt；
- EOS token；
- prompt 与 completion 的边界。

改变 chat template 可能同时改变 SFT loss、GRPO 格式 reward 和最终评估结果，因此它是实验合同的一部分。

### 3.3 Open-R1 当前 SFT 数据语义

官方 recipe 使用 `open-r1/Mixture-of-Thoughts`，核心字段为：

```text
messages: list[{role, content}]
num_tokens: int
source: string
```

在固定依赖 TRL 0.18.0 中：

```text
messages
-> apply_chat_template
-> text
-> tokenize
-> input_ids / attention_mask / labels
-> causal LM loss
```

该数据属于 language-modeling 格式。`completion_only_loss` 默认值为 `None` 时，会对整个非 padding 序列计算 loss。

### 3.4 Full-sequence 与 assistant-only 的目标差异

令上下文 `c = system + user`，assistant 回复为 `y`：

```text
assistant-only:  L = -log p(y | c)
full-sequence:   L = -log p(c, y) = -log p(c) - log p(y | c)
```

两者都能训练 assistant 回复中的推理过程；只要 `<think>...</think>` 位于 assistant 内容中，assistant-only 并不会丢掉 reasoning loss。

区别在于：

- assistant-only 只把 assistant token 设为监督目标，但 system/user token 仍作为上下文参与 assistant token 的预测。
- full-sequence 还直接训练模型预测 system/user/role token，可能带来额外的领域适应信号，也可能稀释有限算力下用于回答部分的梯度。
- 当前 Open-R1 baseline 保留官方 full-sequence 语义；assistant-only 可以在 baseline 通过后作为显式消融，不能静默替换。

### 3.5 `input_ids`、`attention_mask` 与 `labels` 不承担同一职责

| 位置 | `input_ids` | `attention_mask` | MiniMind assistant-only `labels` | Open-R1 full-sequence `labels` |
|---|---|---:|---|---|
| user token | user 的 token id | 1 | `-100` | user 的 token id |
| assistant token | assistant 的 token id | 1 | assistant 的 token id | assistant 的 token id |
| PAD token | pad token id | 0 | `-100` | `-100` |

- `input_ids` 决定模型实际读到哪些 token。
- `attention_mask` 决定哪些位置是有效上下文，哪些是 padding。
- `labels == -100` 决定哪些位置不直接计入 loss。

因此，assistant-only 中的 user token 虽然 `label=-100`，但它没有从输入中消失：`attention_mask=1`，仍会间接影响后续 assistant token 的条件概率。

特殊 token 也不会因为“特殊”而自动屏蔽。EOS 通常应作为有效 label，让模型学会停止；`<think>`、`</think>`、`<answer>`、`</answer>` 如果已经进入序列，通常也是正常训练目标。

### 3.6 决定 loss token 的两道过滤器

只读 `open_r1/sft.py` 不能最终确定哪些 token 参与 loss，因为真正的数据整理发生在固定版本的 TRL 中。需要依次检查：

1. **保留过滤器**：chat template、tokenization 和 `max_length` 决定哪些 token 进入最终序列。
2. **监督过滤器**：`labels` 与 `-100` mask 决定保留下来的 token 中，哪些位置直接贡献 loss。

不能把“样本 tokenized 后没有被截断”和“该 token 参与 loss”混为一件事。

### 3.7 截断为什么会改变训练语义

例：样本共 1000 token，前 300 个是 prompt，后 700 个是 assistant 回复。

| 设置 | 截断前有效 loss token | 右截断至 800 后有效 loss token |
|---|---:|---:|
| assistant-only | 700 | 500 |
| full-sequence | 1000 | 800 |

右截断丢掉的是 assistant 回复末尾 200 token，可能包含后半段推理、`</think>`、最终答案、`</answer>` 和 EOS。若长样本占比高，模型会反复看到不完整回答，可能出现推理收不住、答案缺失、格式不闭合和 EOS rate 下降。

正式训练前不能只信数据集已有的 `num_tokens`，必须使用目标 tokenizer 和最终 chat template 实测：

- token 长度的 P50 / P90 / P95 / P99；
- 总截断率；
- assistant token 保留比例；
- 最终答案、闭合标签和 EOS 的保留率。

### 3.8 与 MiniMind 最重要的差异

MiniMind 的 `SFTDataset.generate_labels()` 显式寻找 assistant 区间，只让 assistant token 参与 loss；当前 Open-R1 recipe 默认采用 full-sequence language-modeling loss。

这不是简单的代码风格差异，而是训练目标差异。第一条 baseline 应先保持官方语义，不擅自改成 assistant-only。

### 3.9 SFT 数据流

```text
dataset row
-> get_dataset()
-> tokenizer / chat template
-> SFTTrainer 数据预处理
-> input_ids + attention_mask + labels
-> model forward
-> masked causal LM loss
-> backward / optimizer
-> checkpoint
```

正式训练前必须拿一条样本检查：

- 模板后的完整文本；
- BOS/EOS/PAD id；
- token 数与截断位置；
- labels 中 `-100` 的位置；
- reasoning 和最终答案是否被截断。

---

## 4. 第三节：GRPO 的核心知识

### 4.1 同一道题生成多个回答

对每个 prompt，策略生成 `G` 个 completions。每个回答分别计算 reward：

```text
r_i = sum_j w_j * r_ij
```

当前数学 demo 主要使用：

- `accuracy`：用 `math-verify` 验证答案。
- `format`：检查 `<think>` 和 `<answer>` 的严格结构。
- `tag_count`：检查四个标签是否完整且只出现一次。

### 4.2 组内相对优势

```text
A_i = r_i - mean(r_group)

当 scale_rewards=True：
A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)
```

如果同一道题的所有回答 reward 一样，组内标准差接近零，模型就缺少“哪个回答更好”的相对信号。

### 4.3 Policy、old policy 与 reference policy

| 对象 | 作用 | 是否更新 |
|---|---|---|
| policy | 当前接受训练的模型 | 是 |
| old policy log-probs | 计算采样前后概率比 | 当前更新期间固定 |
| reference policy | 提供 KL 约束 | 否；`beta=0` 时可能省略 |
| reward functions | 把 completion 转成标量反馈 | 否 |

### 4.4 GRPO loss 直觉

```text
rho_t = exp(log pi_theta - log pi_old)
```

TRL 使用裁剪后的 policy objective，避免一次更新让新策略偏离采样策略太远；当 `beta != 0` 时，再加入相对 reference policy 的 KL 惩罚。loss 只在 completion token 上聚合。

### 4.5 GRPO 数据流

官方数学 demo 使用 `open-r1/OpenR1-Math-220k`：

```text
problem
-> conversation prompt
-> chat template
-> 每题生成 G 个 completions
-> accuracy / format / tag rewards
-> group mean / std
-> advantage
-> old/current/reference log-probs
-> clipped policy loss + optional KL
-> optimizer
-> policy checkpoint
```

其中 `problem` 是 prompt，`solution` 是 accuracy reward 使用的可验证 ground truth。

### 4.6 重点监控信号

- 各 reward 的 mean/std；
- zero-std group 比例；
- completion length 和截断比例；
- EOS rate；
- KL 与 clip ratio；
- loss、梯度是否 NaN；
- format reward 上升但 accuracy 不变的 reward hacking。

---

## 5. 第四节：Open-R1 源码地图

### 5.1 固定源码身份

| 项目 | 固定值 |
|---|---|
| 本地目录 | `D:\pythonlearning\open-r1` |
| commit | `1416fa0cf21595d2083b399a2a0bbddd7f6e9563` |
| Open-R1 | `0.1.0.dev0` |
| TRL | `0.18.0` |
| Transformers | `4.52.3` |
| PyTorch | `2.6.0` |

### 5.2 入口与职责

| 文件 | 主要职责 |
|---|---|
| `src/open_r1/sft.py` | 组装数据、模型、tokenizer、SFTTrainer、保存和恢复 |
| `src/open_r1/grpo.py` | 构造 prompt、选择 reward、组装 GRPOTrainer |
| `src/open_r1/rewards.py` | accuracy、format、tag、code 等 reward registry |
| `src/open_r1/configs.py` | dataset、SFT、GRPO 参数 dataclass |
| `src/open_r1/utils/data.py` | dataset 或 dataset mixture 加载 |
| `src/open_r1/utils/model_utils.py` | AutoTokenizer 和 AutoModel 加载 |
| `scripts/run_benchmarks.py` | benchmark 参数入口，主要面向 Slurm helper |

### 5.3 关键抽象边界

Open-R1 的两个训练入口都比较薄：

- `sft.py` 不直接实现 labels、collator 和训练循环，这些由 `trl.SFTTrainer` 接管。
- `grpo.py` 不直接实现 rollout、advantage 和 GRPO loss，这些由 `trl.GRPOTrainer` 接管。

所以源码阅读必须包含 Open-R1 wrapper 和固定版本 TRL，不能只看两个入口文件。

`sft.py` 自己能够确认的控制流是：

```text
解析参数与 seed
-> checkpoint 检查
-> get_dataset()
-> get_tokenizer() / get_model()
-> chat template fallback
-> 构造 SFTTrainer
-> train()
-> 保存 metrics / state / model
-> 可选 evaluate()
```

而 `apply_chat_template`、tokenization、labels、causal LM loss、backward 和 optimizer step 的具体实现，需要继续进入固定版本的 TRL、Transformers 与 Accelerate 核对。

### 5.4 Chat template 的选择优先级

当前源码链路是：

```text
YAML / CLI 显式传入的 training_args.chat_template
    > tokenizer 自带的 chat_template
    > 两者都没有时，由 setup_chat_format(..., format="chatml") 提供 fallback
```

`get_dataset()` 只负责加载/整理数据，并不会替 tokenizer 决定模板。模板还必须与 EOS 和 generation config 对齐，否则可能出现生成不停止、轮次边界错误、格式 reward 失败或答案抽取失败。保存前，Open-R1 会把 `generation_config.eos_token_id` 对齐到 `tokenizer.eos_token_id`。

### 5.5 评估入口

官方 README 对单 GPU 给出的实际入口是 LightEval CLI，例如：

```text
lighteval vllm MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    --output-dir OUTPUT_DIR
```

`utils/evaluation.py` 中的 helper 主要用于 Slurm 集群，不应直接作为单张 T4 的本地入口。

`trainer.evaluate()` 与 LightEval 回答的问题不同：

- `trainer.evaluate()` 在配置的 eval split 上运行训练体系中的 generation、reward 与 GRPO loss，主要用于训练诊断。
- LightEval 把保存后的 checkpoint 当作独立待测模型，在 MATH-500、AIME 等标准 benchmark 上按固定生成与评分合同计算 accuracy / pass@1。
- 总 eval reward 上升可能只来自 format/tag reward；GRPO eval loss 也随当前采样、advantage 与 probability ratio 改变，因此两者都不能替代标准 benchmark 分数。

评估中的三个概念不能混淆：

- 多次采样估计 pass@1：同题独立生成 N 个完整回答，用样本正确率估计单次生成的正确概率。
- pass@k：k 个完整回答中至少一个正确。
- top-k sampling：每一个生成位置只从概率最高的 k 个 token 中采样，属于 decoding 参数，不是 pass@k。

B0/B1/B2 必须固定 benchmark、chat template、system prompt、temperature、top-p、max tokens、每题响应数、seed policy、答案提取和指标聚合方式。

---

## 6. 第五节：MiniMind 与 Open-R1 对照

| 维度 | MiniMind | Open-R1 |
|---|---|---|
| 模型 | 自定义 MiniMind 模型 | Hugging Face `AutoModelForCausalLM` |
| 训练循环 | 手写 forward/backward/step | TRL Trainer + Accelerate |
| SFT labels | assistant-only mask | 当前 recipe 默认 full-sequence LM |
| padding | dataset 中固定长度 padding | collator 动态 padding |
| 配置 | argparse | dataclass + YAML + CLI override |
| checkpoint | 自定义 `.pth` | Hugging Face checkpoint 目录 |
| GRPO rollout | Torch/SGLang 自定义引擎 | TRL Transformers/vLLM |
| GRPO loss | MiniMind 源码中显式实现 | TRL 内部实现 |
| reward | 启发式 + reward model | 可验证数学/代码规则 registry |

可以迁移：loss mask、advantage、KL、completion mask、checkpoint 和失败诊断等概念。

不能直接假设：两个项目的 labels、GRPO 初始化、reference model、loss_type、reward 和保存协议相同。

---

## 7. 第六节：当前复现合同

### 7.1 已冻结

- source commit：`1416fa0cf21595d2083b399a2a0bbddd7f6e9563`。
- base model：`Qwen/Qwen3-0.6B-Base`。
- SFT 入口：`src/open_r1/sft.py`。
- GRPO 入口：`src/open_r1/grpo.py`。
- B2 初始化：从本地 B1 SFT checkpoint 加载模型与 tokenizer，并以新的 GRPO optimizer/run 启动，不恢复 SFT optimizer 状态。
- 对照对象：B0、B1、B2。
- 环境方式：Python 3.11 + `uv`。
- 硬件目标：单张 T4 16GB，执行前重新审计。
- 声明边界：T4 缩放复现，不匹配官方 7B/多 H100 绝对分数。

### 7.2 数据候选

SFT：

```text
open-r1/Mixture-of-Thoughts
config: math
split: train
subset: 固定 seed shuffle 后 select 前 N 条
```

GRPO：

```text
open-r1/OpenR1-Math-220k
config: default
split: train
prompt: problem
ground truth: solution
rewards: accuracy + format + tag_count
```

### 7.3 评估候选

- 主指标：MATH-500 pass@1 / accuracy。
- 次指标：AIME 2024 pass@1。
- 行为指标：format rate、EOS rate、平均输出 token、截断比例。
- 效率指标：latency、throughput、peak VRAM。
- 训练诊断：loss、reward mean/std、zero-std group、KL、clip ratio。

B0/B1/B2 必须固定相同的 chat template、system prompt、生成参数、采样次数、seed 和答案提取逻辑。

### 7.4 已冻结与 Week 2 测量门

- [x] B2 明确从本地 B1 checkpoint 初始化；保留 B1，不覆盖其目录。
- [x] SFT 使用 `math` config；首条 baseline 聚焦数学推理，不声明通用能力。
- [x] 子集选择固定为数据版本/fingerprint + `shuffle(seed=42)` + 前 N 条；N 由 Week 2 throughput 决定。
- [ ] Week 2 在 B0 canonical eval 前写回 MATH-500 响应数与最终生成参数；首选上游每题 4 个响应，T4 不可行则三模型统一降为 1 并记录偏差。
- [ ] Week 2 打印真实 tokenizer/sample 后固定 Qwen3 chat template、EOS、PAD，不猜 token id。

---

## 8. 本周学习和源码任务

- [x] 阅读 Open-R1 README 的 Overview、Installation、Training、Evaluation、Data Generation。
- [x] 阅读 DeepSeek-R1 报告中 distillation、R1-Zero、multi-stage training 部分。
- [x] 固定 source commit。
- [x] 建立源码入口地图。
- [x] 追踪 `sft.py` 到 TRL SFTTrainer 边界。
- [x] 区分 full-sequence 与 assistant-only 的训练目标。
- [x] 区分 `input_ids`、`attention_mask`、`labels` 和 `-100` mask。
- [x] 理解 max length 截断对 reasoning、最终答案和 EOS 的影响。
- [x] 确认 chat template 的覆盖优先级与 EOS 对齐要求。
- [x] 学习并追踪 `grpo.py`、reward 到 TRL GRPOTrainer 边界。
- [x] 学习并确认单 GPU evaluation 为什么使用 LightEval CLI，以及它与 `trainer.evaluate()` 的边界。
- [x] 完成 GRPO 单样本数据流、组内 reward、advantage、概率比、clipping、KL 与 loss reduction 的口头自测。
- [x] 冻结数据、指标、seed policy，以及生成参数的测量门和 fallback。
- [x] 更新 `PLAN.md`，将其设为唯一权威复现合同。

### 8.1 已完成的口头学习检查（2026-07-20）

- [x] 能解释 B0、B1、B2 的职责与实验归因。
- [x] 能说明为什么当前选 SFT → GRPO，同时不把 SFT 误认为理论必需。
- [x] 能计算 assistant-only 与 full-sequence 的有效 loss token 数。
- [x] 能解释被 mask 的 prompt 仍如何参与 assistant token 的条件生成。
- [x] 能识别长回复右截断对最终答案和训练质量的风险。
- [x] 能说明为什么只读 `sft.py` 不能确定最终 loss mask。
- [x] 已纠正“chat template 在 dataset 加载时自动决定”和“tokenizer 原模板总是优先”两个误区。
- [x] 能解释 `Dataset.map()` 新增 `prompt` 后为何仍保留 `solution`，并区分 Open-R1 wrapper、reward registry 与 TRL `GRPOTrainer` 的职责。
- [x] 能从一条 GRPO prompt 追踪到 completions、rewards、advantage、current/old/reference log-probs 与 scalar policy loss。
- [x] 能区分训练内 eval、LightEval benchmark、pass@1、pass@k 与 top-k sampling。

## 9. 学习自测

1. B0、B1、B2 分别解决什么归因问题？
2. 为什么 chat template 不只是展示格式？
3. MiniMind assistant-only labels 与 Open-R1 full-sequence LM loss 有什么区别？
4. 一个 GRPO prompt 为什么要生成多个 completions？
5. 同组 reward 全部相同时会发生什么？
6. Open-R1 的 GRPO loss 真正实现在哪里？
7. 为什么 `trainer.evaluate()` 不等同于 MATH-500 reasoning accuracy？
8. 哪些 T4 缩放会改变实验语义，必须单独记录？

## 10. 阻塞与进度

- 当前 phase：Week 1 `analysis` 已完成；下一 phase 为 Week 2 `setup`。
- 当前 trust：`unverified`。
- 已运行训练：否。
- 已接受 baseline：否。
- 环境兼容性尚未审计。
- Week 1 的 SFT/GRPO 数据流、loss contract、截断风险、源码边界与 LightEval 边界均已完成学习检查。
- B2 初始化、SFT 数据域、子集规则、主指标和 seed policy 已冻结；硬件相关数值已有明确的 Week 2 测量门。
- 下一锚点：Week 2 环境审计、真实样本检查与 B0 baseline 准备。

## 11. 权威来源

- [Open-R1 pinned source](https://github.com/huggingface/open-r1/tree/1416fa0cf21595d2083b399a2a0bbddd7f6e9563)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [TRL 0.18.0 SFTTrainer](https://github.com/huggingface/trl/blob/v0.18.0/trl/trainer/sft_trainer.py)
- [TRL 0.18.0 GRPOTrainer](https://github.com/huggingface/trl/blob/v0.18.0/trl/trainer/grpo_trainer.py)
- [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
