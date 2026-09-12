# Gate 0：独立 SFT 受控实验合同

> 状态（2026-09-12）：Gate 0A、数据人工复核、B0、首 batch shadow、20-step smoke 与 TRL 100-step 恢复 pilot 已完成；正式 S1 与训练后配对评测仍为 `NO-GO`。S1 启动前的剩余硬门见 [`S1_PRELAUNCH_PLAN.md`](S1_PRELAUNCH_PLAN.md)。
>
> 规则：先写目标、假设和验收标准，再看正式训练结果。`TBD` 未关闭前，不启动真实模型正式训练。
>
> 分阶段实施、项目结构、本机/GPU 边界和产物要求见 [`REAL_SFT_EXECUTION_PLAN.md`](REAL_SFT_EXECUTION_PLAN.md)。本文是结果可信度合同，执行计划不得覆盖本文中的冻结变量。

## 1. 实验身份

| 字段 | 当前值 |
| --- | --- |
| 实验名称 | Qwen3-0.6B-Base 数学推理独立 SFT 受控实验 |
| 创建日期 | 2026-08-23 |
| 负责人 | 学习者本人 |
| 代码版本 | `TBD` |
| 合同状态 | 数据、B0 与工程 pilot 证据已完成；正式 S1 与训练后配对评测尚未批准 |

## 2. 第一组问题：目标与假设

这一组由学习者先独立填写。描述必须指向可观察行为，不能只写“让模型变好”或“学习 SFT”。

### 2.1 想改变模型的什么行为

给定一条具有可验证最终答案的数学问题，希望 `Qwen3-0.6B-Base` 经过 SFT 后，更稳定地生成符合约定格式、能够被答案解析器提取、且最终答案正确的回答。

本实验首先观察三种行为：是否正常结束并生成 EOS、最终答案是否可被解析、解析后的最终答案是否通过确定性验证。推理过程的主观质量只作为辅助分析，不作为第一个实验的主要成功标准。

填写提示：模型目前在什么输入上表现出什么问题；训练后希望哪一种输出行为发生变化。

### 2.2 当前 baseline 的已知表现

历史实验中，学习者记得 `Qwen3-0.6B-Base` 在 MATH-500 上的结果约为 `10%`，但原始结果文件、运行命令、逐样本输出和完整配置已经找不到，因此该数字只作为历史记忆值，不能作为正式对照证据，也不能写入最终实验结论。

正式实验必须重新执行 B0：使用未训练的固定基座，在完整 MATH-500 上按冻结后的评测合同运行，保存完整命令、环境与代码版本、generation config、汇总指标和逐样本输出。B0 与 S1 必须复用同一评测合同。

填写提示：只写有证据的位置、指标或样本；未经归档的 Open-R1 分数标记为“待核验”，不能直接作为可信结论。

### 2.3 可被证伪的实验假设

本实验区分三层假设，避免把“实现正确”和“榜单提升”混为一谈：

1. **H1，核心实现假设**：独立实现的 masked SFT loss、梯度和单步参数更新与固定 PyTorch 参考计算数值对齐；tiny dataset 可以被过拟合。H1 不成立时，不进入真实模型效果归因。
2. **H2，学习行为假设**：在固定模型、数据切分、prompt/chat template 和训练配置后，经过验证的数学示范 SFT 会降低同分布 held-out completion token loss，并提高或维持答案可解析率和正常 EOS 率。
3. **H3，任务能力研究问题**：在 B0/S1 使用完全相同评测合同且完成数据泄漏检查的前提下，观察 MATH-500 等数学任务的最终答案正确率及错误类型如何变化。H3 暂不预设必然提升；未提升不自动等于 SFT 实现错误。

此外，训练前先建立 Base 模型的通用能力画像；训练后复用同一通用评测面板检查回归。专项数学能力提高但通用能力明显下降时，不能把实验简单判为成功。

建议句式：

> 在固定模型、数据切分、prompt/chat template、推理参数和评测器的情况下，使用 ______ 数据进行 SFT，将使 ______ 指标或错误类型发生 ______ 变化；如果出现 ______，则当前假设不成立或证据不足。

### 2.4 为什么当前首先选择 SFT

当前目标是从预训练 Base 模型出发，使用可信的标准回答和推理示范，让模型学习目标条件分布中的回答方式、任务模式、输出格式与风格。SFT 直接对目标 completion token 做最大似然训练，既适合这一监督信号，也是后续 token log-prob、mask、梯度聚合和训练工程实现的基础验证载体。

DPO 需要 chosen/rejected 偏好对来优化相对偏好；GRPO 需要同一 prompt 的多次在线采样、可验证 reward 和组内优势。它们解决的反馈类型不同，不应为了学习顺序而强行用于当前任务。对“Base 模型变成能按要求回答的数学助手”这一实验，先完成 SFT 是合理的工程顺序；但 SFT 不是分类、embedding、继续预训练或所有模型用途的普遍必经步骤。

填写提示：现有反馈是否是可信的标准回答或示范轨迹；为什么不应只用 prompt、DPO 或 GRPO 解决。

## 3. 模型与更新方式

| 决策 | 当前值 | 依据 |
| --- | --- | --- |
| 最小正确性载体 | 手工 tiny tensors；随后使用固定随机种子的 `nn.Embedding -> nn.Linear` tiny causal LM | 先隔离公式、shape、mask 和梯度，不引入真实模型复杂性 |
| 真实实验基座 | `Qwen/Qwen3-0.6B-Base` | 官方 0.6B 预训练 causal LM；沿用 Open-R1 经验 |
| model revision | `311c62e88814bff7206909ccd330bab0a784743b` | 锁定当前官方仓库版本，避免 `main` 漂移 |
| 主实验更新方式 | FP32 主参数与 AdamW 状态、BF16 autocast 的全参数 SFT | 保留 FP32 更新语义并降低前向/反向激活与矩阵计算成本；目标是验证完整参数更新链路，不用 LoRA 规避主链路 |
| LoRA / QLoRA | 不进入主实验；后续作为参数高效训练对照 | 避免首次实验同时引入 adapter、量化和不同可训练参数集合 |
| tokenizer 版本 | 与真实基座使用相同 repo 和 revision | 词表、special token 与 checkpoint 严格匹配 |
| chat template | 使用上述 revision 内置的官方 Qwen ChatML template | 不复制或自行改写模板；保存渲染后文本和 token ids 作为证据 |
| 文档/填充 token | `<|endoftext|>`，ID `151643` | Qwen 的 eod token；padding 位置仍由 attention mask 和 label mask 排除 |
| 对话轮次结束 token | `<|im_end|>`，ID `151645` | Qwen eot token；SFT 与生成停止条件均显式对齐到该 token |

### 3.1 模板使用合同

- B0 与 S1 的生成输入都使用同一 tokenizer、同一 chat template 和 `add_generation_prompt=True`。
- SFT 样本使用完整 user/assistant 对话渲染，不附加新的 generation prompt。
- prompt、system、user、padding 和 assistant header token 的 label 设为 `-100`；只训练 assistant completion 及其轮次结束 token `<|im_end|>`。
- 不把 padding token 是否等于 EOS 当作 mask 依据；有效训练位置只由显式 labels mask 决定。
- 保存 tokenizer special token 映射、chat template 内容哈希、渲染文本、`input_ids`、`labels` 和 assistant mask 的样例审计。

## 4. 数据合同

| 项目 | 当前值 |
| --- | --- |
| 原始数据 | `open-r1/OpenR1-Math-220k` / `default` / `train` |
| 原始数据 revision | `e4e141ec9dea9f8326f4d347be56105859b2bd68` |
| 训练 artifact | 候选已生成：62,212 条，SHA-256 `de1d09abae28a9af1117e1677e74a4c62d417ea4a8670af293f6263803a4c18b`；人工复核前不标记最终 Frozen |
| 验证 artifact | 2,000 条，SHA-256 `18b56115fd332838edecd9606cd67fd67cdcf5a50280a2a07339d796424cb8c6` |
| 测试数据 | 完整 MATH-500；通用回归面板在评测合同冻结时补齐版本 |
| split 规则 | 先过滤、去重和去污染；再按稳定题目键的 SHA-256 排序，前 2,000 条为 validation，其余为 train |
| split salt | `independent-sft-v1` |
| 去重 | 对规范化 problem 做 SHA-256 精确去重；缺失或无效 UUID 不影响稳定主键；近重复扫描结果必须归档 |
| 评测泄漏检查 | 对 MATH-500、GSM8K 和通用回归任务执行规范化精确匹配与 pinned Open-R1/s1 whitespace word 8-gram overlap；命中样本在 split 前删除 |
| 最大长度 | `16384`；由 A100-80GB 最长样本训练探针确定（22,295-token 样本失败、16,384-token 通过），冻结为 `gate0b-16k-derived-a`（实测 `max 16,384`、`p99 15,626`、`p50 4,863`） |
| 截断策略 | 不静默截断 reasoning 或最终答案；超过模型合同长度的样本从本实验排除并单独报告数量、来源和长度分布 |
| prompt/assistant mask | prompt/system/user/assistant header/padding 为 `-100`；仅 assistant completion 和结尾 `<|im_end|>` 参与 loss |

必须回答：被截断内容是否改变答案；训练数据是否包含测试题或近重复题；是否只训练 assistant/completion token。

### 4.1 原始行到 SFT artifact 的过滤规则

对每个原始题目执行以下确定性流程，任何一步失败都写入 rejected manifest 并记录原因码：

1. `problem` 非空，`messages` 恰好为一轮 `user -> assistant`，assistant 内容非空。
2. assistant 内容必须与 `generations[i]` 中某一条精确对应，保存该 generation index。
3. 仅保留 `correctness_math_verify[i] == true`；只通过 Llama judge 的样本不进入第一次受控实验。
4. 要求 `is_reasoning_complete[i] == true`，并检查 `<think>` / `</think>` 闭合和最终 `\boxed{}` 可提取。
5. 使用 `problem` 构造 user message，使用通过校验的 `generations[i]` 构造 assistant message，不直接信任简化后的 `messages` 字段。
6. 稳定题目键为 `sha256(normalize(problem))`；原始 `uuid` 仅作为 provenance 保存，因为数据中存在无效的 `uuid="NaN"`。
7. 保存原始 row index、source、uuid、generation index、验证字段、规范化题目哈希、渲染后 token 数和完整样本内容哈希。

### 4.2 数据源审查结论

- `OpenR1-Math-220k/default` 有约 93.7k 个题目，来自 NuminaMath 1.5，每题包含 2 到 4 条 DeepSeek-R1 推理轨迹；官方说明多数使用 Math Verify，另有一部分使用 Llama judge。
- `Mixture-of-Thoughts/math` 的 93,733 条是上述 default 子集的简化 SFT 视图，字段只有 `messages`、`num_tokens` 和 `source`。它适合参考官方 SFT 配方，但缺少本实验所需的逐 generation 验证和 provenance 字段，因此不作为新的冻结原始数据源。
- 对原始 default 数据首部和分层 offset 共抽查 500 行：500/500 的 assistant 可对应到某条 generation；348/500 由 Math Verify 通过，152/500 仅由 Llama judge 通过；499/500 标记为 reasoning complete；500/500 含闭合 `</think>` 和 `\boxed`；发现无效 `uuid="NaN"`。这只支持“数据可进入完整过滤”，不能替代全量扫描。
- 数据集为英文、Apache-2.0。固定 revision 的全量扫描已产出合格条数、精确重复和 pinned Open-R1 word 8-gram 污染证据；人工抽样仍未完成，因此数据合同尚未最终 Frozen。

### 4.3 全量自动审计结果

2026-09-04 对固定的 93,733 条原始数据按 pinned Open-R1/s1 whitespace word 8-gram 规则完成两次独立全量运行。除 `created_at` 外，两份 manifest 完全相同：

- accepted `64,212` 条，其中 train `62,212`、validation `2,000`；rejected `29,521` 条。
- train/validation/rejected/review SHA-256 分别为 `de1d09ab...4c18b`、`18b56115...cb8c6`、`e23d2ad1...3a253`、`1c2e1846...b56645`。
- accepted sequence token 数的 `min/p50/p95/p99/max` 为 `105/4,940/14,084/16,850/22,295`，没有样本超过 `32,768`。
- 主要拒绝计数为 `math_verify_failed=28,627`、`MATH-500 8-gram overlap=509`、`incomplete_reasoning=188`、`MMLU 8-gram overlap=165` 和 `duplicate_problem=46`。一条记录可以包含多个原因，因此原因计数总和不要求等于 rejected 行数。
- 自动抽样文件包含 20 条 accepted 和 27 条 rejected（9 类原因各 3 条）。这只是待复核清单，不等于人工复核已经通过。
- 正则去标点切分与 Open-R1 whitespace 切分在 1,301 个原始 problem 上作出不同污染判断，因此旧 artifact 被否决；最终规则的来源和全量兼容性对照保存在 `evidence/decontamination_method_review.json`。

### 4.4 旧 1,000 条本地 artifact 的信任等级

旧文件仅标记为 `reference_only`，不得直接作为新实验训练输入：

- `sft_manifest.json` 只保存来源、角色和长度，没有 dataset revision、稳定样本 ID、题目内容哈希或验证字段；它记录的 source 是 `OpenR1-Math-220k`，但旧 collator 审计记录的数据集是 `Mixture-of-Thoughts/math`。
- `sft_sample_ids.json` 实际只含 `seed=42` 和 `count=1000`，没有任何样本 ID。
- 旧选择方式是 streaming shuffle buffer，正式训练配置也未消费 manifest，无法证明训练时使用了同一批样本。
- 旧长度审计使用未锁 revision 的 tokenizer；`eos_at_end_pct=0` 是因为它把 `<|endoftext|>` 当作 EOS，而对话实际以 `<|im_end|>` 结束，不能解释为样本没有结束标记。
- 旧 `assistant_retencion_ratio` 实际计算的是 `max_len / total_length`，不是 assistant token 或最终答案保留率。
- 旧 collator 使用 full-sequence labels，user token 也参与 loss，与本实验 assistant-only 合同不一致。

## 5. 评测合同

评测分为三层，B0 与 S1 必须复用相同任务版本、prompt/few-shot 口径、generation config、解析器和 scorer：

1. **通用能力画像与回归**：使用固定版本 LightEval 的 MMLU、ARC-Challenge 和 HellaSwag 检查知识、推理与常识能力是否发生明显遗忘。这三项都是 log-likelihood 多选评测，不受随机采样和回答格式变化干扰。
2. **专项数学能力**：使用完整 MATH-500 作为主任务，GSM8K 作为补充任务，并从最终答案正确率、可解析率、正常 EOS 率和错误类型四个角度比较 B0/S1。
3. **训练语义与实现正确性**：held-out completion token loss、PyTorch 参考数值对齐、tiny overfit 和单步更新对照，不用下游榜单替代核心计算验证。

| 项目 | 当前值 |
| --- | --- |
| 主要指标 | H1 数值/梯度/单步更新对齐；held-out completion token loss；MATH-500 最终答案正确率 |
| 辅助指标 | 答案可解析率、正常 EOS 率、输出长度与逐样本错误类型 |
| 回归指标 | MMLU 5-shot、ARC-Challenge 25-shot、HellaSwag 10-shot；分别报告，不合成为单一“通用分” |
| verifier / answer parser | LightEval commit `d3da6b9bbf38104c8b5e1acc86f83541f9a502d1` + `math-verify==0.5.2` |
| 逐样本错误分类 | `correct`、`unparseable`、`wrong_answer`、`no_eos`、`length_truncated`、`runtime_error`；允许一个样本同时带多个健康标签 |
| 评测随机种子 | vLLM 与 LightEval 固定 `1234`；few-shot 选择也固定同一 seed；B0/S1 不得改变 |

### 5.1 评测软件栈

- 参考入口锁定 OpenR1 commit `1416fa0cf21595d2083b399a2a0bbddd7f6e9563` 的评测方法，不跟随 `main` 漂移。
- LightEval 锁定 commit `d3da6b9bbf38104c8b5e1acc86f83541f9a502d1`；数学验证锁定 `math-verify==0.5.2`。
- 服务器首次安装时保存完整 lock 文件、Python/CUDA/driver/GPU 信息、`pip freeze` 和评测代码 commit。
- 数据集物化已经保存 Hugging Face revision、split、行数和 Arrow fingerprint。Gate 0 仍保持 Draft 的原因是人工数据复核和服务器实验未完成，而不是缺少数据身份。

### 5.2 专项数学评测

#### MATH-500 主评测

| 字段 | 冻结值 |
| --- | --- |
| LightEval task | `lighteval|math_500|0|0` |
| 数据 | `HuggingFaceH4/MATH-500` / `test` / 完整 500 题 |
| prompt | 固定 Qwen chat template，`add_generation_prompt=True`，不额外加入 system prompt |
| dtype | BF16 |
| `max_model_length` | `32768` |
| `max_new_tokens` | `32768` |
| sampling | `temperature=0.6`、`top_p=0.95`、seed `1234` |
| 每题采样数 | 4；由任务的 `math_pass_at_1_4n` 指标触发 |
| 主报告 | `math_pass@1:1_samples` 与 `math_pass@1:4_samples`，两者均保留 |
| 停止 token | `<|im_end|>`；同时记录是否因长度上限停止 |

采用这组参数是为了与 OpenR1 已公开的 MATH-500 评测口径对齐，而不是在看到结果后调温度或输出长度。历史约 `10%` 的记忆值不参与比较；必须重新运行 B0。

#### GSM8K 补充评测

| 字段 | 冻结值 |
| --- | --- |
| LightEval task | `leaderboard|gsm8k|4|0` |
| 数据 | `gsm8k` / `main` / `test` |
| few-shot | 4，沿用 Qwen Base 历史公开评测口径 |
| generation | greedy，`temperature=0`，`max_new_tokens=256` |
| 指标 | LightEval `quasi_exact_match_gsm8k` |

MATH-500 和 GSM8K 必须分开启动，因为两者 generation config 不同。不得为了复用一条命令而给 GSM8K 使用 MATH-500 的随机采样配置。

### 5.3 通用能力回归面板

| 能力 | LightEval task | 口径 | 指标性质 |
| --- | --- | --- | --- |
| 跨学科知识 | 固定版 `examples/tasks/open_llm_leaderboard_tasks.txt` 中全部 `leaderboard|mmlu:*|5|0` | 57 个 subject，5-shot | log-likelihood accuracy |
| 科学推理 | `leaderboard|arc:challenge|25|0` | 25-shot | normalized log-likelihood accuracy |
| 常识续写 | `leaderboard|hellaswag|10|0` | 10-shot | normalized log-likelihood accuracy |

- 这些任务不调用 chat template，不生成开放式回答；直接比较候选 continuation 的条件 log-prob，目的是测量预训练能力是否被 SFT 遗忘。
- 每项分别报告官方指标、正确题数、总题数和 B0/S1 差值，不把不同任务粗暴平均成一个分数。
- 中文和代码能力确实属于完整能力画像，但固定版 LightEval 没有与上述任务同等直接的 C-Eval 合同，而代码执行评测还需要独立 sandbox。它们作为 Gate 1 扩展面板，不阻塞第一次受控 SFT；文档不得把当前面板表述为“全部通用能力”。

### 5.4 输出健康与错误分析

对 MATH-500、GSM8K 和 validation generation 保存逐样本记录，至少包含：

- 数据集 revision、sample id、原始 problem 和最终渲染 prompt 的哈希。
- 原始输出文本、output token ids、生成 token 数、停止原因和是否生成 `<|im_end|>`。
- parser 提取值、gold value、verifier 结果和异常信息。
- `correct`、`unparseable`、`wrong_answer`、`no_eos`、`length_truncated`、`runtime_error` 标签。
- B0/S1 同题配对结果，用于统计“错变对”“对变错”和格式变化，而不只保存总分。

其中 `unparseable` 表示没有得到可交给 verifier 的最终答案；`wrong_answer` 表示成功解析但验证失败。两者必须区分，否则无法判断 SFT 改善的是格式还是数学能力。

### 5.5 变化与回归判据

- 对确定性任务保存逐题正确性，并对 B0/S1 的配对差值执行按题目 bootstrap，报告 95% confidence interval。
- 对 MATH-500 按题目聚合四次采样结果后做配对 bootstrap；同时报告 1-sample 和 4-sample 指标，不能只挑更好看的一个。
- 某项通用任务同时满足“绝对下降至少 1 个百分点”且“配对差值的 95% CI 上界小于 0”时，记为明确回归。
- 绝对下降达到 2 个百分点但置信区间仍跨 0 时，记为回归告警，需要复跑或扩展分析，不能宣称无退化。
- 指标变化小且置信区间跨 0 时，只能写“没有足够证据证明变化”，不能写“能力完全不变”。
- H2 的核心证据是固定 validation 上 assistant-only completion token NLL 下降，同时 EOS 率和可解析率不恶化；H3 的 MATH-500 变化单独作为任务研究结果。

至少保存：输入、原始输出、解析结果、样本分数、错误类别和运行配置。LLM judge 不能作为第一个确定性任务的唯一 scorer。

## 6. 控制变量与实验组

| 实验 | checkpoint | 用途 | 训练变化 |
| --- | --- | --- | --- |
| B0 | `Qwen/Qwen3-0.6B-Base@311c62e88814bff7206909ccd330bab0a784743b` | 训练前 baseline | 无 |
| S1 | B0 经**固定版本 Open-R1/TRL 训练栈**完成本合同 SFT 后的 checkpoint | **主实验**；H2/H3 的主证据 | 仅进行本合同定义的 SFT 更新，1 个 epoch |
| S1-ind | B0 经**独立实现 runner**完成同一合同 SFT 后的 checkpoint | 实现级同条件对照；验证自研实现与固定 TRL 栈在完整 epoch 尺度上语义一致 | 与 S1 相同的 train artifact、样本顺序、effective batch、loss mask、optimizer 与 scheduler；**不作为 H2/H3 的主证据** |
| R1a | B0，不保存新 checkpoint | PyTorch/TRL 首 batch shadow 数值对照 | 同一 batch 只做 forward/backward，不 step |
| R1b | B0 经固定参考框架的短程 checkpoint | 100 个 optimizer step 的双实现工程对照（TRL 与自研 runner 同条件） | 使用与 S1 相同样本顺序和优化语义，不要求完成整轮训练 |

### 6.1 S1 与 S1-ind 共用的优化合同

下表同时适用于 S1（固定 TRL 栈）与 S1-ind（独立实现 runner）。两者的差异只允许出现在实现拓扑上，不允许出现在下列冻结值上。

| 字段 | 冻结值 |
| --- | --- |
| objective | assistant-only causal cross-entropy；按整个 optimizer step 内的有效 assistant token 总数求平均 |
| precision | FP32 主参数、梯度与 AdamW moments；矩阵计算位于 BF16 autocast；loss numerator 与 token count 使用稳定归约类型 |
| optimizer | AdamW：`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.0`；首次实验不引入参数分组差异 |
| learning rate | `4e-5` |
| scheduler | cosine with minimum learning-rate ratio `0.1` |
| warmup | 总 optimizer steps 的 `3%`，按 Transformers `TrainingArguments.get_warmup_steps` 向上取整；16K 冻结 artifact 有 61,224 条训练记录，S1 共 479 steps，因此是 15 steps |
| max grad norm | `0.2` |
| epochs | 1 个完整 train split epoch |
| effective batch | 每个完整 optimizer window 为 128 条 sequence；S1 单轮最后一个 window 是剩余 4 条，不跨 epoch 填充；不同长度的有效 token 数另外记录 |
| packing | false |
| max length | `16384`；与冻结 artifact `gate0b-16k-derived-a` 一致（实测 `max 16,384`）。超长样本已按数据合同排除，不截断 |
| train seed | `42` |
| sample order | 对稳定样本键做固定 seed shuffle；恢复训练必须恢复 sampler/RNG 状态 |

这里采用 OpenR1 的公开 SFT 配方作为优化器和 scheduler 起点，但只训练 1 epoch，并保持本实验自己的 assistant-only loss 和数据过滤合同。`4e-5` 是预注册起始值，不代表它已经是 Qwen3-0.6B 的最优学习率。

### 6.2 与 pinned Open-R1 / TRL 的实现关系

本实验追求的是核心训练语义有公开参考、差异可解释，而不是声称复现 Open-R1-Distill-7B 的最终 checkpoint。对照版本固定为 Open-R1 `1416fa0...`、TRL `0.18.0` 和 Transformers `4.52.3`。

自 2026-09-11 起 **S1 使用固定版本的 TRL 训练栈**（见 §6 与 §11 变更记录），因此下表的"本实验"列指 **S1 的单卡 TRL 运行**。表中以"独立计算""独立实现"描述的路径属于自研 runner，它在 S1-ind 中按同一合同运行，用于验证两侧在完整 epoch 尺度上语义一致；其数值对齐证据由 H1 与首 batch shadow（R1a）承担。

| 项目 | pinned 参考实现 | 本实验 | 判定 |
| --- | --- | --- | --- |
| causal LM loss | Transformers 模型接收 `labels` 和 accumulation window 的 `num_items_in_batch` | 独立计算 loss sum，并除以同一 window 的全部有效 token | 数值与参数更新测试对齐 |
| label mask | TRL `DataCollatorForLanguageModeling(completion_only_loss=True)` 支持 prompt-completion mask | user、header、padding 为 `-100`，仅 assistant completion 与 `<|im_end|>` 监督 | 与 TRL completion-only collator 测试对齐 |
| Open-R1 公开 SFT loss | `messages` 数据且 `completion_only_loss=None`，TRL 0.18 默认训练完整 sequence | assistant-only | 有意不同；避免把 prompt token 纳入本实验目标 |
| chat template | Open-R1-Distill-7B 使用自定义长 system prompt | 固定 Qwen3-0.6B-Base revision 自带模板，不新增 system prompt | 有意不同；B0/S1 必须一致 |
| optimizer | `adamw_torch`，betas `0.9/0.999`、eps `1e-8`、weight decay `0` | `torch.optim.AdamW` 同参数 | 对齐 |
| scheduler | cosine with min LR、warmup ratio `0.03` | 同类型；warmup 使用 Transformers 的 `ceil` 规则 | 对齐 |
| effective batch | 8 GPUs x 2 sequences x accumulation 8 = 128 | 1 GPU x 1 sequence x accumulation 128 = 128，末尾 remainder 除外 | 完整 window 对齐 |
| precision / checkpointing | BF16 compute；Open-R1 的 BF16 参数由 ZeRO-3 管理高精度 master/optimizer state；gradient checkpointing 使用 `use_reentrant=False` | 单卡 FP32 主参数与 AdamW state、BF16 autocast、相同 gradient checkpointing 设置 | 计算精度与优化状态语义对齐，实现拓扑不同 |
| kernel / topology | ZeRO-3、Liger、8 GPU | 单设备、FlashAttention 2、无 Liger | 有意不同；不改变 loss 合同 |
| epoch / data | Mixture-of-Thoughts、5 epochs | 经过审计的 OpenR1-Math 子集、1 epoch | 有意不同；这是受控实验而非模型复刻 |
| checkpoint export | Open-R1 保存 tokenizer，写入 generation EOS，最终恢复 `use_cache=True` | 每个完整 checkpoint 保存 tokenizer、generation EOS 和推理态 `use_cache=True`，加载训练时再关闭 cache | 对齐且可直接被 LightEval 加载 |

上述“对齐”必须由测试或真实运行证据支持；表中的“有意不同”是预注册变量，不能在看到结果后临时改成 Open-R1 配方。

### 6.3 固定变量与硬件适配变量

B0、S1 和后续同条件复跑必须固定：model/tokenizer revision、训练 artifact hash、split、样本顺序、chat template、label mask、effective batch、优化器语义、学习率、scheduler、epoch、评测任务、prompt、scorer 和 generation config。

Gate 0B 当前 runner 是明确的单设备实现，因此本轮 S1 冻结为单 GPU、micro-batch `1`、gradient accumulation `128`。不得把尚未实现和验证的 DDP/ZeRO/FSDP 当作可调开关；若最长样本探针无法通过，应更换显存更大的单卡实例，或另开实验合同实现并验证分布式路径。checkpoint 保存频率可因磁盘与任务 wall-time 调整，但必须记录，且不得通过降低 `max_length`、改数据或改 loss 来掩盖 OOM。

## 7. 正确性与工程验收

### 7.1 层 A：算法正确性

第一个实现单元固定为：

```python
def masked_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    ...
```

函数合同：

- `logits` 必须为浮点 tensor，shape 为 `[B, T, V]`；`labels` 必须为整数 tensor，shape 为 `[B, T]`，且前两维完全一致。
- 先做 causal shift：`logits[:, :-1, :]` 预测 `labels[:, 1:]`；第一个 label 和最后一个 logit 不参与这一组 next-token 配对。
- `ignore_index` 位置不进入 loss numerator，也不进入 denominator；最终 scalar 是整个 batch 内全部有效 token loss 的平均值，不是先按样本平均再按 batch 平均。
- `T < 2`、shape/dtype 不符合合同或 shift 后没有任何有效监督 token 时抛出带明确信息的 `ValueError`，不返回静默的 0 或 NaN。
- 不原地修改 `logits` 或 `labels`；返回值必须保留到有效 `logits` 的 autograd 路径。

- [x] causal shift 测试。
- [x] `ignore_index=-100` mask 测试。
- [x] 不同有效长度测试。
- [x] dtype 边界合同。
- [x] 全 mask 边界合同。
- [x] 与 PyTorch 参考 loss 数值对齐。
- [x] backward 梯度有限且流向正确。
- [x] tiny dataset 过拟合与自回归复现测试。

### 7.2 层 B：真实模型

- [ ] smoke run 无 NaN/Inf。
- [ ] 有效 token、loss、grad norm 和 learning rate 可追踪。
- [ ] checkpoint 可加载、可生成；EOS 行为按相对 B0 的对照记录（smoke 阶段只记录不判定，见 §11 变更记录）。
- [ ] 与固定版本参考框架比较首 batch loss、梯度或单步更新。
- [ ] B0/S1 使用同一评测合同。
- [ ] 保存逐样本输出和错误分析。

B0/S1 LightEval 结果必须通过 `scripts/compare_lighteval_results.py` 配对。比较器先验证 evaluation contract hash、任务、chat-template 模式、生成参数、prompt、few-shot、输入 token、gold 和 choices 完全相同，再统计每项 metric 的提升/退化/持平数量，并把所有预测变化与退化样本写入 `changed_samples.jsonl`。因此聚合分数相同也不能掩盖样本级“有进有退”。

## 8. 资源与停止条件

| 项目 | 当前值 |
| --- | --- |
| 本机正确性测试预算 | CPU，单次全套测试目标不超过 10 分钟；不下载 0.6B checkpoint 也能完成 H1 |
| 服务器数据 gate | 全量扫描固定 revision 的 93,733 个原始题目，生成 accepted/rejected manifests、长度分布、去重和去污染报告 |
| 服务器 smoke 预算 | 128 条固定样本、20 个 optimizer step；在继续正式训练前检查 loss、梯度、显存和 checkpoint round-trip |
| B0 预算 | 正式训练前完成第 5 节全部 baseline，并保存逐样本结果；不得用历史约 10% 代替 |
| 正式训练预算 | 全部合格 train artifact，1 epoch；不在同一次 run 临时追加 epoch |
| 硬件要求 | 支持 BF16 的 GPU；具体型号、数量、显存和并行策略在租用实例后写入 run manifest |
| 硬停止条件 | loss/grad/参数出现 NaN 或 Inf；样本、mask、tokenizer 或 checkpoint hash 不匹配；恢复后 sampler/RNG 不连续；评测集污染 gate 未通过 |
| 软停止条件 | 已积累至少 20 step 基线后，连续 3 个 optimizer step 的 loss 高于此前 20 step 中位数 5 倍，或 pre-clip grad norm 高于此前 20 step 中位数 10 倍；暂停并保留现场，不自动调参继续。普通 gradient clipping 只记录，不构成停止条件 |

正式训练前必须依次通过本机正确性测试、服务器全量数据 gate、B0 和 smoke。达到停止条件时先保存配置、最后一个正常 step、异常 batch 的稳定 ID、loss、grad norm、RNG/sampler 状态和 traceback，再决定是否生成新的合同版本；不得原地改配置并覆盖失败记录。

## 9. 结果判定

### 支持假设

分层判定，不用一个总分掩盖问题：

1. **H1 支持**：手工样例成立；loss 与 PyTorch 参考在 FP32 下满足 `rtol=1e-5, atol=1e-6`；关键参数梯度满足 `rtol=1e-4, atol=1e-6`；单步参数更新对齐；tiny dataset 的有效 token NLL 可降至 `0.05` 以下且生成结果可复现训练样本。
2. **H2 支持**：S1 在固定 validation 上的逐样本 assistant-only completion NLL 相对 B0 下降，配对 bootstrap 的 95% CI 上界小于 0；EOS 率与可解析率均未下降超过 2 个百分点。
3. **H3 得到正向证据**：MATH-500 或 GSM8K 的配对差值为正，且 95% CI 下界大于 0。若只提高格式/解析率但最终答案指标没有可信变化，应报告为“格式改善”，不能报告为“数学能力提升”。
4. **Gate 0 主实验通过**：H1、H2 均成立，checkpoint round-trip 通过，且通用回归面板没有第 5.5 节定义的明确回归。H3 是否正向单独报告，不作为证明核心 SFT 实现正确的必要条件。

H1 的对象是**独立实现本身**，由 tiny 测试、PyTorch 参考对齐和首 batch shadow（R1a）判定，不依赖 S1。H2 与 H3 在 **S1（固定 TRL 栈）** 上判定。S1-ind 用于回答"自研实现与固定 TRL 栈在同一合同下是否语义一致"；若两者差异超出冻结容差，应先按 §6.2 定位实现差异，不得挑选更有利的一侧作为结论。

### 否定假设

- H1 任一数值、梯度、更新或 tiny overfit 合同失败，否定当前独立实现正确性，停止真实效果归因。
- H1 成立但 held-out completion NLL 没有下降，或 EOS/可解析率明显下降，则当前数据与 SFT 行为假设 H2 不受支持。
- 输出更符合格式但 MATH-500/GSM8K 没有可信提升，H3 的能力提升主张不受支持；这不自动否定 H1。
- 专项任务提高但任一通用任务出现明确回归，只能报告 trade-off，不能把 S1 定义为整体优于 B0。

### 实验无效或证据不足

- B0/S1 使用了不同 model/tokenizer revision、prompt、chat template、scorer、generation config 或测试数据。
- 正式训练数据未完成固定 revision 全量扫描、去重和评测去污染，或 artifact/hash 无法追溯。
- 训练中途改 learning rate、epoch、max length、mask 或数据后仍沿用同一实验 ID。
- 只有汇总分数而没有逐样本输出，或只有训练 loss 而没有固定 validation/B0。
- 评测随机性未固定、结果无法复跑，或置信区间太宽而仍声称“提升/无退化”。
- 发生 runtime error、输出被长度截断或 parser 故障但未单独报告。

实现错误、数据泄漏、评测器错误、配置不可追溯和关键变量不一致，应判为实验无效，而不是算法失败。

## 10. Gate 0 冻结检查

- [x] 第一组问题已经由学习者填写并通过审核。
- [x] 模型、数据 revision 和 split 规则已冻结；候选 artifact hash 已完成双跑复现。
- [x] 主要指标、回归指标和 verifier 已冻结。
- [x] B0/S1 的固定变量已经列出。
- [x] tiny、smoke 和正式实验预算已确定。
- [x] 支持、否定和无效三类判定已经区分。
- [ ] accepted/rejected 人工抽样完成。
- [ ] 合同状态由 Draft 改为 Frozen，并记录冻结日期与 commit。

## 11. 变更记录

| 日期 | 状态 | 变化 | 原因 |
| --- | --- | --- | --- |
| 2026-08-23 | Draft | 创建 Gate 0 骨架 | 启动独立 SFT 受控实验 |
| 2026-08-23 | Draft | 冻结候选模型、原始数据过滤、评测、优化和结果判定合同 | 在写训练代码前消除数据泄漏、变量混杂和事后挑指标风险；等待服务器数据 fingerprint 与运行环境关闭最后 gate |
| 2026-08-23 | Gate 0A reviewed | 允许开始不依赖真实 checkpoint/数据的 masked SFT loss 与 tiny tests | 本机核心计算不应被服务器数据下载阻塞；真实 S1 仍须关闭 Gate 0B |
| 2026-08-27 | Gate 0A implementation in progress | loss/梯度/单步更新、tiny overfit 与自回归复现主链通过；保留 dtype 与不同有效长度合同 | 只按已执行证据更新状态，不把未覆盖边界写成完成 |
| 2026-08-27 | Gate 0A complete | dtype、不同有效长度与全部 H1 本机验收证据通过 | 15 条 CPU 测试覆盖 loss、mask、聚合、梯度、SGD 更新、tiny overfit 与自回归复现；Gate 0B 仍保持 Draft |
| 2026-08-29 | Local optimizer step complete | gradient accumulation、冻结参数与 gradient clipping 通过 gold tests | 22 条 CPU 测试证明不等有效 token micro-batch 与完整 batch 更新对齐；可观测性和混合精度后续单独验收 |
| 2026-08-31 | Local SFT data contract complete | right-padding collator 与 assistant-only labels 通过测试 | 32 条 CPU 测试覆盖 padding、attention mask、assistant label mask、输入不可变和无监督样本拒绝；真实 tokenizer/template 集成仍待验收 |
| 2026-09-01 | Pinned tokenizer integration complete | 单轮 Qwen chat template 到 assistant-only labels 的适配通过 | 37 条默认单元测试与 1 条独立集成测试通过；确认锁定模板无 generation block、prompt 是 full conversation 前缀、`<|im_end|>` 参与监督且尾部换行被 mask |
| 2026-09-02 | Local SFT vertical slice complete | 结构化 batch 到 HF 风格 forward、独立 loss 和 optimizer update 接通 | 40 条默认测试与 1 条 tokenizer 集成测试通过；保留全局 token 归一化、冻结参数和 clipping 不变量；真实权重 smoke 与 TRL 对照仍属 Gate 0B |
| 2026-09-03 | Local Readiness Gate complete | 允许开始全量数据冻结，之后进入受限服务器会话 | 82 条默认测试、1 条集成测试、TRL/Transformers 参考对齐、真实数据重复审计，以及真实 Qwen3-0.6B BF16 两步训练与跨进程恢复通过；B0、服务器 smoke 和 S1 尚未开始 |
| 2026-09-04 | Full automated data audit superseded | 固定 93,733 条原始数据，首次生成并双跑复现 train/validation/rejected/review artifact | 后续发现正则去标点切分与 pinned Open-R1 不一致，因此该组 artifact 和 bundle 被否决 |
| 2026-09-04 | Open-R1 decontamination alignment | 将正则去标点 8-gram 修正为 pinned Open-R1/s1 whitespace word 8-gram，并重新完成两次全量审计 | 旧规则在 1,301 行上与参考实现不同，因此旧 artifact/bundle 被否决；新 artifact 双跑哈希一致，人工抽样仍待完成 |
| 2026-09-04 | Open-R1/TRL implementation audit | 修正 warmup 向上取整，补齐 tokenizer/EOS/use-cache checkpoint 合同，并冻结单设备拓扑与最长样本显存门禁 | 将参考实现一致项、实验有意差异和真实工程缺陷分开，避免把“参考 Open-R1”误写成逐项复制 |
| 2026-09-11 | 服务器 smoke 生成门校正 | 明确 EOS 率只按相对 B0 判定，smoke 阶段不设绝对 EOS 要求；§7.2 对应条目改为"记录并对照"；生成门脚本的 `--stop-tokens` 与 `--attn-implementation` 参数化 | greedy 下 B0 与 20 步 SFT 均 `0/4` EOS；合同采样配置（`temperature=0.6, top_p=0.95`）下分别为 `0/8` 与 `1/8`，SFT 相对 B0 提升 `12.5` 个百分点，而 B0 输出完全退化。终止信号只占监督 token 的 `0.017%`；原阶段 E 门要求 `eos_rate=1.0`，不可达且严于 §9 H2。证据：`evidence/gate0b-16k-generation-gate-diagnosis.json`、`evidence/gate0b-16k-generation-gate-sampled-comparison.json` |
| 2026-09-11 | 实验组重新定义：S1 归属固定 TRL 栈 | §6 实验组表：S1 改为固定版本 Open-R1/TRL 训练栈，新增 S1-ind 作为自研 runner 的同条件对照；§6.1 标题改为 S1/S1-ind 共用合同；§6.2 明确"本实验"列指单卡 TRL 运行；§9 明确 H1 由 tiny 测试与首 batch shadow 判定、H2/H3 在 S1 上判定 | 解决 `REAL_SFT_EXECUTION_PLAN.md` §6.1 记录的合同冲突：计划已决定正式能力实验优先使用锁定版本的 Open-R1/TRL、自研 runner 作为语义显微镜与小规模验证路径，而原合同仍把自研 runner 的 1 epoch 定义为 S1 |
| 2026-09-11 | 训练最大长度按实测冻结为 16384 | §4 数据合同与 §6.1 优化合同的 `max length` 由过时的 `32768` 改为 `16384`；§5.2 评测生成预算（`max_model_length` / `max_new_tokens` = `32768`）**保持不变** | 实测冻结 artifact `gate0b-16k-derived-a` 序列长度为 `max 16,384 / p99 15,626 / p50 4,863`；该上限由 A100-80GB 显存探针确定（22,295 失败、16,384 通过）。§5.2 的 `32768` 是生成预算而非训练长度，且与 Open-R1 公开 MATH-500 口径对齐，B0/S1 必须一致 |
