# Gate 0B：真实 SFT 工程实验执行计划

> 状态：In progress；阶段 A 与阶段 B 自动审计已完成，阶段 B 人工复核和阶段 C-H 尚未完成
>
> 本文只回答“如何实施”。模型、数据、指标、控制变量和结果判定以 [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md) 为准；执行过程中不得用本文覆盖实验合同。

## 1. 本阶段目标

使用冻结版本的 `Qwen3-0.6B-Base` 和经过审计的 OpenR1-Math 样本，亲手搭建一条不依赖 `Trainer` 的真实 SFT 训练链路，并以 Open-R1/TRL 为参考实现完成数值对照、训练前后同条件评测、断点恢复和证据归档。

完成后应当能够回答：

1. 一条原始数学样本如何变成只监督 assistant completion 的 `input_ids`、`attention_mask` 和 `labels`。
2. 一批不同长度样本如何经过模型、loss、梯度累积、裁剪、优化器和 scheduler 更新权重。
3. checkpoint 除模型权重外还需要保存哪些训练状态，恢复后为什么能继续同一次实验。
4. 如何证明训练实现正确、模型确实学到训练分布，以及专项能力是否提升、原有能力是否回归。

## 2. 实现边界

### 2.1 由我们实现和维护

- 实验配置校验、随机种子和 resolved config 归档。
- 原始数据审计、过滤、去重、去污染、稳定切分和 manifest。
- chat template 适配、assistant-only labels、collator 和 DataLoader。
- 原生 PyTorch 训练循环、有效 token 全局归约和 gradient accumulation。
- BF16 autocast、gradient clipping、AdamW、warmup 和 cosine scheduler 接线。
- train/validation 指标、NaN/Inf 硬停止、日志和运行摘要。
- 模型、optimizer、scheduler、sampler/RNG 和进度状态的 checkpoint/resume。
- 首 batch shadow 对照、smoke、正式训练和训练后错误分析的编排。

### 2.2 使用成熟库但必须理解接口合同

- `transformers`：Qwen 架构、预训练权重、tokenizer 和 `generate`。
- `torch`：tensor 算子、autograd、AdamW、AMP 和分布式底层能力。
- `datasets`：固定 revision 数据的读取和物化。
- `accelerate`：只有单卡链路通过后，才允许用于设备或多卡基础设施；不得替代 loss 和训练语义。
- `LightEval`、`vLLM`、`math-verify`：正式评测和确定性答案验证。

### 2.3 只作参考，不进入主训练路径

- Open-R1 的 `sft.py` 和固定版本 TRL `SFTTrainer`。
- torchtune 的 recipe 分层和 checkpoint 设计。

参考实现用于发现差异，不是正确性的唯一来源。发生差异时先比较数据、模板、labels、有效 token 归约、optimizer 和 scheduler 语义，再决定哪一侧有问题。

## 3. 代码与产物布局

文件只在对应阶段开始时创建，不预建空模块。目标布局如下：

```text
independent_implementation/
├── EXPERIMENT_CONTRACT.md
├── REAL_SFT_EXECUTION_PLAN.md
├── configs/
│   └── gate0b/
│       ├── data_audit.yaml
│       ├── b0_eval.yaml
│       ├── sft_smoke.yaml
│       └── sft_train.yaml
├── scripts/
│   ├── audit_openr1_math.py
│   ├── evaluate_checkpoint.py
│   ├── compare_first_batch.py
│   └── train_sft.py
├── src/post_training_core/
│   ├── data.py
│   ├── sft.py
│   ├── training.py
│   ├── experiment.py
│   ├── checkpointing.py
│   └── observability.py
├── tests/
│   ├── fixtures/
│   ├── test_data_audit.py
│   ├── test_checkpointing.py
│   ├── test_observability.py
│   └── ...已有 SFT 测试
└── runs/                       # 本地生成，不提交大型内容
    └── gate0b/<run_id>/
```

每个真实 run 至少生成：

```text
runs/gate0b/<run_id>/
├── run_manifest.json
├── config.resolved.yaml
├── environment.json
├── data_manifest.json
├── metrics.jsonl
├── checkpoints/
├── evals/
└── report.md
```

Git 保存代码、配置、测试、轻量 manifest、汇总指标和报告；原始数据、模型权重、完整逐 token tensor 和大型逐样本输出放在外部 artifact 目录，只在 Git 中保存路径、hash 和生成命令。

## 4. 执行阶段和门禁

### 阶段 A：本机工程骨架

**运行位置**：Windows 本机，CPU；现有 8 GB GPU 不是前置条件。

**实现内容**：

- 配置读取、字段校验、路径解析和 resolved config。
- run directory 和 manifest 创建。
- 统一随机种子入口。
- 指标 JSONL writer、异常现场保存和退出码。
- checkpoint payload 的最小结构与 round-trip 测试。
- 使用缩小的 `Qwen3Config` 实例化随机 `Qwen3ForCausalLM`，通过真实 Qwen 模型类跑通数据、forward、backward、optimizer、scheduler 和 checkpoint/resume。
- 使用冻结的真实 Qwen tokenizer 和少量真实 OpenR1-Math 样本跑通同一条 CLI；只缩小模型尺寸和样本数，不替换数据字段、模板、labels 或训练语义。
- CLI 支持显式的 `--dry-run`、`--max-steps`、`--limit-samples` 和 `--device` 调试参数，所有 override 必须写入 resolved config，不能形成隐藏的第二条代码路径。

**验收门**：

- 全部现有单元测试继续通过。
- 新模块均可用 tiny model、tiny dataset 在 CPU 测试。
- 不下载真实权重也能验证配置、日志和 checkpoint 状态合同。
- 同一个 `train_sft.py` 入口能够在本机完成 2-step 训练、保存、启动新进程恢复并继续到 step 4。
- 对空监督 batch、NaN loss、无效路径、损坏 checkpoint 和配置不一致执行失败注入，验证程序会在更新权重前停止并留下可诊断记录。

#### Local Readiness Gate

下面的检查全部通过并生成 `local_readiness_report.json` 后，才允许创建付费 GPU 服务器：

- [x] lint/静态检查和全部默认单元测试通过。
- [x] 冻结 Qwen tokenizer 集成测试通过。
- [x] tiny Qwen CPU 端到端训练与参数变化测试通过。
- [x] tiny Qwen checkpoint/resume 与连续运行结果在冻结容差内一致。
- [x] 少量真实 OpenR1-Math 样本能够完成审计、物化、collate 和训练。
- [x] 数据审计在重复运行后得到相同 hash。
- [x] 评测入口能够用固定 fixture 验证 scorer、错误分类和输出文件 schema。
- [x] CLI 正常路径和关键失败路径的 subprocess 测试通过。
- [x] `uv lock`、启动命令和服务器环境清单已经冻结。
- [x] 预估显存、磁盘、下载量、step 时间和最长允许运行时间已经记录。

2026-09-04 已重新生成当前 [`evidence/local_readiness_report.json`](evidence/local_readiness_report.json)，状态为 `passed: true`。本机使用真实 `Qwen3-0.6B-Base`、真实审计样本、BF16 参数和梯度检查点完成两个独立进程的 step 1 保存与 step 2 恢复；源码树、训练集和 `uv.lock` hash 均已绑定。sequence length 为 `105`，两步峰值显存均约 `5.73 GB`，两个进程各约 `110` 秒；两个 checkpoint 合计约 `6.69 GiB`。这些数据证明当前代码的低显存兼容性、保存和恢复链路，不替代正式 FP32 主参数路径的服务器最长样本 memory probe、B0、smoke 或效果评测。

### 阶段 B：数据审计与冻结

**运行位置**：先用本机 fixture 和少量真实样本调试；全量扫描可在本机 CPU 或服务器 CPU 执行，不占用 GPU 训练时间。

**数据流**：

```text
固定 revision 原始行
-> 外层 schema 校验
-> generation 与验证字段配对
-> Math Verify / reasoning 完整性过滤
-> problem 规范化和精确去重
-> MATH-500 / GSM8K 污染检查
-> chat template 渲染和 token 长度统计
-> 超长样本隔离，不静默截断
-> 稳定 SHA-256 切分
-> train / validation artifact + rejected manifest
```

**必须产出**：原始 fingerprint、accepted/rejected 数量、reason code 分布、长度分布、重复与污染报告、train/validation 样本 hash。

**验收门**：同一输入重复运行产生相同样本顺序和 hash；人工抽查至少 20 条 accepted 和每类 rejected 样本；合同中的模型、数据版本和 split 项由 Draft 改为 Frozen。

#### 阶段 B 当前证据

- [x] 固定 `OpenR1-Math-220k@e4e141e...` 的 93,733 条原始行完成两次独立全量扫描。
- [x] 两次运行均得到 train `62,212`、validation `2,000`、rejected `29,521` 和 review `47` 条。
- [x] train、validation、rejected、review 四个 artifact 的 SHA-256 在两次运行中全部一致。
- [x] accepted sequence 的 `p50/p95/p99/max` 分别为 `4,940/14,084/16,850/22,295` tokens；全部低于冻结上限 `32,768`。
- [x] 原始数据、五个评测集、tokenizer revision、Arrow fingerprint、chat template hash、special token 映射和运行时版本已写入 manifest。
- [ ] 学习者或指定审查者逐条复核 20 条 accepted 和按 9 类原因抽取的 27 条 rejected 记录，并记录结论。
- [x] 主实验严格采用 pinned Open-R1/s1 的 whitespace word 8-gram 污染合同，不把未预注册的模糊匹配算法混入本次复现；方法对照记录保存在 `evidence/decontamination_method_review.json`。

人工复核使用同一条可恢复命令，不直接修改 `review_samples.jsonl`：

```powershell
uv run python scripts/review_audit_samples.py `
  --review-samples E:\small_model_post_training_gate0b\artifacts\gate0b-full-openr1-a\review_samples.jsonl `
  --data-manifest E:\small_model_post_training_gate0b\artifacts\gate0b-full-openr1-a\data_manifest.json `
  --output evidence\data_review_decisions.jsonl `
  --reviewer learner
```

对 accepted 选择 `keep/flag`，对 rejected 选择 `agree/disagree`。每次判断后都会原子保存，退出后执行同一命令从未完成项继续。需要修改旧判断时增加可重复的 `--redo <屏幕显示编号>`。只有 47 条全部完成且没有 `flag/disagree` 时汇总状态才是 `passed`；有争议项时先解决，不通过重跑覆盖决定。server bundle 构建器强制读取该 summary，校验 decision 文件、review artifact hash、完成数和未解决数，无法再靠文档约定绕过这道门。

人工判断使用下面的固定口径，不能因为最终答案碰巧正确就忽略低质量推理：

- accepted 只有在题目文本足以独立作答、推理连贯且支撑最终答案、没有伪造搜索/引用或无依据猜测、没有明显自相矛盾与异常重复、输出结构适合作为 SFT 示范时才选 `keep`；任一项不满足选 `flag` 并在 comment 写明可观察证据。
- rejected 只判断记录展示的 `review_reason` 是否确实存在；存在选 `agree`，规则误判或证据不足选 `disagree`，并写明哪条事实与自动判定冲突。
- `flag/disagree` 不是直接改成 `keep/agree` 来关闭。先判断这是单条异常还是系统性过滤缺口；若需要修改审计规则，必须重新双跑全量审计、重新抽样和重新完成人工复核。

### 阶段 C：B0 训练前基线

**运行位置**：GPU 服务器。只做推理和评测，不更新参数。

**执行内容**：

- validation assistant-only completion NLL。
- MATH-500 与 GSM8K 专项评测。
- MMLU、ARC-Challenge、HellaSwag 通用能力画像。
- EOS、可解析率、输出长度、停止原因和逐样本错误记录。

**验收门**：所有任务保存完整命令、版本、配置、逐样本输出和汇总指标；历史记忆中的约 `10%` 或 Open-R1 旧报告不能替代 B0。

### 阶段 D：首 batch shadow 对照

**运行位置**：GPU 服务器，不执行 `optimizer.step()`。

使用同一批已经物化的 `input_ids`、`attention_mask` 和 `labels` 分别运行：

```text
我们的 PyTorch 路径 -> loss_sum / valid_tokens / mean_loss / grads
固定 TRL 参考路径  -> loss_sum / valid_tokens / mean_loss / grads
```

优先比较有效 token 数、FP32 mean loss、选定参数梯度和 grad norm。BF16 不要求逐 bit 相等，但差异必须落入冻结容差并能解释。

**验收门**：数据和 labels 完全一致；loss/梯度对齐通过；没有 step，因此 base checkpoint 不被修改。

### 阶段 E：20-step smoke

**运行位置**：GPU 服务器；固定 128 条样本。

验证完整路径：

```text
load checkpoint
-> DataLoader
-> BF16 forward
-> assistant-only loss
-> gradient accumulation
-> unscale / clip
-> optimizer.step
-> scheduler.step
-> metrics
-> save checkpoint
-> fresh process reload
-> validation loss + generate
```

**验收门**：无 NaN/Inf；参数确实变化；loss、有效 token、grad norm、learning rate 和显存可追踪；checkpoint 在新进程可加载、继续训练并正常生成 `<|im_end|>`。

服务器首次启动仍属于兼容性验证，不属于正式训练。它只处理本机无法覆盖的真实权重、Linux/CUDA、BF16、最长 22,295-token 样本的训练显存和实际吞吐问题。本轮 runner 明确为单设备实现，不把未经实现级验证的 NCCL/多卡路径算作可用能力；发现普通 Python 逻辑或测试可提前覆盖的问题时，立即停止实例，回到本机修复并重新通过 Local Readiness Gate。

### 阶段 F：100-step pilot 与恢复测试

**运行位置**：GPU 服务器；使用正式 train artifact 和正式配置。

- 从 B0 独立启动，训练到 step 50 后正常退出。
- 从 checkpoint 恢复并训练到 step 100。
- 运行固定 validation NLL 和小型 generation panel。
- 使用固定 TRL 配置完成 100-step 参考运行，只比较实现语义和趋势，不追求最终最好分数。

**验收门**：恢复后的 global step、样本顺序、optimizer、scheduler 和 RNG 连续；validation NLL 有合理变化；硬停止条件未触发。失败时保留现场，禁止原地改配置后覆盖 run。

### 阶段 G：正式 S1

**运行位置**：GPU 服务器。

- 从 B0 权重重新启动，不从 smoke 或参考框架 checkpoint 继续。
- 使用冻结后的全部 train artifact 完成合同规定的 1 epoch。
- 不在看到结果后临时修改数据、mask、学习率、epoch 或评测口径。
- 中间 checkpoint 只用于容错恢复，不作为事后挑选最好结果的候选池。

**验收门**：完成最终 checkpoint、训练日志、validation NLL、环境与 artifact hash 归档；checkpoint round-trip 再次通过。

### 阶段 H：S1 训练后评测与报告

**运行位置**：GPU 服务器评测；本机汇总和审查。

完全复用 B0 合同，输出：

- B0/S1 train-distribution validation NLL 对照。
- MATH-500、GSM8K、MMLU、ARC-Challenge、HellaSwag 对照。
- 同题“错变对”“对变错”、格式改善、无 EOS、截断和运行错误分类。
- 配对 bootstrap 95% confidence interval。
- H1、H2、H3 分层结论，以及支持、否定或证据不足判定。

## 5. 本机与服务器的职责边界

| 工作 | 本机 | GPU 服务器 |
| --- | --- | --- |
| 核心公式、mask、梯度测试 | 必须 | 复跑 |
| 配置、日志、checkpoint tiny 测试 | 必须 | 复跑 |
| tiny Qwen 同构端到端训练与恢复 | 必须 | 不重复调试 |
| 数据 fixture 与少量真实样本审计 | 必须 | 可复跑 |
| 全量数据审计 | 可执行 | 可执行，不需要 GPU |
| 0.6B 首 batch BF16 shadow | 不要求 | 必须 |
| 20-step smoke / 100-step pilot | 不要求 | 必须 |
| 正式 SFT 与完整评测 | 不执行 | 必须 |

本机 8 GB 显卡已验证真实 0.6B BF16 参数兼容性路径在长度 1726 的单样本上可以完成两步训练与跨进程恢复。正式训练采用 FP32 主参数/AdamW state 与 BF16 autocast，因此该本机结果不代表正式精度路径、22,295-token 最长样本、有效 batch 或完整数据规模可运行；正式配置不得为了迁就本机而改变实验问题。

### 5.1 服务器成本控制

- 数据集、模型和评测依赖优先下载到可复用的持久卷，避免每次租用 GPU 重新下载。
- 服务器启动脚本先校验 git commit、lock、模型 revision、数据 hash、CUDA 和磁盘，再申请正式运行。
- 环境门通过后先对全量 train artifact 中最长的 22,295-token 样本执行一次 forward/backward/optimizer memory probe；失败时不启动 B0 或 smoke。
- 第一次付费运行只执行真实 Qwen 的 one-batch preflight 和 20-step smoke，完成后主动退出，不顺势启动正式 S1。
- smoke 日志和 checkpoint 下载到本机审查；确认后再创建 100-step pilot 任务。
- 100-step pilot 与正式 S1 使用独立 run id 和输出目录，失败 run 永不覆盖。
- 每个任务设置 wall-time、最大 step 和磁盘上限；异常达到硬停止条件后立即保存现场并退出。
- 设置 `save_total_limit=2`，只在新 checkpoint 原子写入成功后清理更旧的完整 checkpoint。
- 正式 S1 只在本机门禁、服务器 smoke、恢复 pilot 和 B0 全部通过后启动。

## 6. 学习与实现方式

每个阶段按同一节奏推进：

1. 先讲清该阶段的输入、输出、状态和失败模式。
2. 冻结一个小接口和对应测试，由学习者先实现。
3. 本机运行测试；一次只定位一个失败。
4. 通过后进行代码审查，确认不是只为测试硬编码。
5. 合并到纵向 smoke，再进入下一阶段。
6. 到 GPU 阶段先执行 preflight，不直接启动正式训练。

这意味着我们不会一次性生成完整训练项目交给学习者阅读，也不会边跑正式训练边补最基本的工程合同。

### 6.1 工程理解与框架对照主线

实验阶段 A-H 负责回答“工程执行到哪里”；下面这条学习主线负责回答“学习者是否真正掌握这条链路”。代码已经存在或测试已经通过，不等于对应能力已经验收。

第一、第二步顺序执行；第三至第五步不是三个彼此分离的大阶段，而是针对每个模块反复完成“理解自己的实现 -> 映射成熟框架 -> 数值或行为对照”，避免看完全部自研代码后再重新理解框架。

#### 第一步：完整工程地图

只梳理入口、调用方向、模块职责和产物，不深入函数内部：

```text
scripts/train_sft.py
-> config / runner / experiment
-> audit / data
-> sft / engine / training
-> checkpointing / observability / provenance
-> evaluation / eval_runner / comparison
-> shadow / trl_reference
```

通过标准：能够不看实现细节，指出配置、数据、模型、训练循环、checkpoint、评测和 TRL 对照分别从哪里进入、由哪个模块负责。

#### 第二步：一条真实样本的端到端数据流

```text
messages
-> tokenizer / chat template
-> input_ids / assistant-only labels / attention_mask
-> collator / batch [B, T]
-> model forward / logits [B, T, V]
-> causal shift / masked token loss
-> backward / gradient accumulation
-> optimizer.step / scheduler.step
-> checkpoint / resume
```

通过标准：能够对一条实际样本解释每一步的输入、输出、shape、设备、dtype、状态变化和失败点，并指出 `scheduler.step()` 只跟随真实 optimizer update，而不是每个 micro-batch。

#### 第三至第五步：逐模块理解、映射和对照

按下面顺序逐块关闭，每块都必须回答“自己的输入输出是什么、TRL/Open-R1 对应在哪里、语义是否一致、什么证据证明”：

| 顺序 | 自研模块 | 成熟框架映射 | 最低对照证据 |
|---:|---|---|---|
| 1 | `audit.py`、`data.py` | Open-R1 数据配置、TRL collator | 同一条样本的 token、labels、mask 和有效 token 数 |
| 2 | `sft.py`、`engine.py`、`training.py` | Transformers causal loss、TRL training step、Accelerate accumulation | 同一 batch 的 loss sum、mean loss、gradient 和参数变化 |
| 3 | optimizer、scheduler、observability | Transformers optimizer/scheduler、Accelerate update 边界 | optimizer step 数、LR、grad norm 和跳步行为 |
| 4 | `checkpointing.py`、`provenance.py`、`readiness.py` | Transformers/Accelerate checkpoint | 保存内容、global step、optimizer/scheduler/RNG 和恢复后的下一步 |
| 5 | `generation.py`、`evaluation.py`、`eval_runner.py` | Open-R1/LightEval/vLLM | 相同生成合同、逐样本输出、汇总指标和错误分类 |

#### 第六步：成熟框架正式实验

学习和 shadow 路径关闭后，正式能力实验优先使用锁定版本的 Open-R1/TRL；自研 runner 作为语义显微镜和小规模验证路径，不以替换成熟训练框架为目标：

```text
server preflight
-> longest-sample memory probe
-> B0
-> first-batch shadow
-> 20-step smoke
-> 100-step pilot + resume
-> 正式 S1
-> B0/S1 配对评测
-> bad case / 能力退化分析
-> 实验结论
```

当前 [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md) 仍把自研 runner 的完整 1 epoch 定义为 S1，并把 TRL 定义为 100-step 参考组。这与本节的最新路线决策冲突。服务器正式训练前必须单独审查并修订实验组、H1/H2/H3 归属和成本预算；在合同修订并重新冻结前，不得用本节直接启动正式 S1。

#### 学习进度

- [x] 第一步：能够讲清完整工程地图。
- [x] 第二步：能够跟踪一条真实样本直到一次 optimizer update 和 checkpoint（2026-09-07，代码与真实记录走读验收；不代表本轮已运行完整 GPU 链路或完成框架数值对照）。
- [ ] 数据模块完成自研、TRL/Open-R1 映射和输出对照。
- [ ] loss、accumulation、optimizer 和 scheduler 完成映射与数值对照。
- [ ] checkpoint/resume 完成状态合同与行为对照。
- [ ] evaluation 完成同条件输出、指标和错误分析对照。
- [ ] 正式训练框架选择已经写入并重新冻结实验合同。
- [ ] B0、smoke、pilot、正式 S1 和配对评测全部形成证据。

## 7. 当前唯一下一步

阶段 A、Local Readiness 和阶段 B 的全量自动审计已通过，去污染方法也已对齐 pinned Open-R1。实现级审计已区分参考实现一致项与本实验的有意差异，并补齐 warmup、可独立加载 checkpoint 和最长样本显存门禁。

当前学习动作只有一个：进入第 6.1 节第三至第五步的第一个模块，从 `audit.py` 的原始记录到冻结样本构造开始，映射锁定版本的 Open-R1/TRL 数据处理路径，并对照同一条样本的 token、labels、mask 和有效 token 数。完成数据模块理解后，再继续阶段 B 的 47 条人工内容复核。

当前实验执行动作仍是阶段 B 人工复核；复核通过后生成最终 server bundle manifest。在租用服务器前，必须先解决“自研 runner 正式 S1”与“Open-R1/TRL 正式 S1”的合同冲突并重新冻结。服务器第一次会话只允许依次执行 `uv sync --frozen`、`scripts/server_preflight.py`、最长样本 memory probe、B0、首 batch shadow 和 20-step smoke，任何门禁失败立即停止，不顺势启动正式 S1。

## 8. 完成定义

只有以下事项全部满足，Gate 0B 才能标记完成：

- [x] 数据 artifact 可追溯、可重建，且两次全量自动审计的全部 artifact hash 一致。
- [ ] 人工抽样复核完成。
- [ ] B0 与 S1 使用完全相同的评测合同。
- [ ] 独立实现与参考实现完成首 batch 数值对照。
- [ ] smoke、恢复训练和正式 1 epoch 均通过。
- [ ] 最终 checkpoint 可重新加载、继续训练和正常生成。
- [ ] H1、H2、H3 分开判定，没有用训练 loss 代替能力评测。
- [ ] 通用能力回归与专项能力变化均有逐样本证据。
- [ ] 代码、配置、环境、日志、hash、失败记录和结论可以由第三方复跑。
