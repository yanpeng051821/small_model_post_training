# 后训练核心算法独立实现

> 当前状态：Gate 0A、Local Readiness 和 Gate 0B 全量自动数据审计已完成；下一步是 47 条审计样本的人工复核与服务器兼容性验证。B0、20-step smoke、100-step pilot 和正式 S1 尚未完成，禁止直接启动正式 S1。

本目录负责不依赖 `Trainer` 封装实现和验证后训练核心计算。Open-R1/TRL 继续作为真实框架与工程对照，不作为本目录的实现模板。

## 1. 当前目标

```text
冻结实验合同
-> masked SFT loss
-> 单卡 SFT update
-> 真实 Qwen SFT 工程与训练前后评测
-> token log-prob
-> DPO loss/update
-> 为后续 GRPO 复用概率、mask 和 reference 组件
```

Gate 0B 的分阶段落地、代码布局、本机/GPU 边界和完成定义见 [`REAL_SFT_EXECUTION_PLAN.md`](REAL_SFT_EXECUTION_PLAN.md)。实验结果是否有效仍以 [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md) 为准。

学习顺序不是生产模型的固定训练顺序。是否在正式任务中采用 DPO、GRPO 或其他方法，仍由任务反馈和受控实验决定。

## 2. 为什么先实现 `masked_sft_loss`

完整训练脚本同时包含模型、数据、优化器、精度、分布式和日志。直接从完整脚本开始，即使 loss 下降，也难以判断下面这些核心语义是否正确：

- causal shift 是否对齐。
- prompt 与 padding 是否被排除。
- assistant token 是否正确参与训练。
- token loss 如何聚合成 scalar loss。
- 全部 label 都被 mask 时如何处理。
- 梯度是否流向预期参数。

因此第一个计算单元暂定为：

```python
def masked_sft_loss(
    logits,
    labels,
    ignore_index=-100,
):
    ...
```

它的验收不是“能够运行”，而是：

1. tiny tensor 手算成立。
2. 与固定 PyTorch 参考计算数值对齐。
3. prompt/padding/masked token 不影响有效 loss。
4. backward 后梯度有限且流向正确。
5. 边界条件有明确合同和测试。

合同冻结前不实现该函数，避免先写代码再反向定义实验目标。

## 3. 两层验证

### 层 A：算法正确性

- tiny tensor 和 tiny model。
- CPU/本机运行。
- 秒级或分钟级。
- 验证公式、shape、mask、不变量和梯度。

### 层 B：真实模型受控实验

- 候选基座：`Qwen3-0.6B-Base`。
- 使用固定数据、评测器和推理合同。
- 与 PyTorch 参考计算及固定版本 TRL/torchtune 做对照。
- 保存配置、日志、checkpoint、逐样本输出和错误分析。

层 A 未通过时不启动层 B。

## 4. 当前动作顺序

- [x] 统一专项进度状态，允许独立实现和预注册实验启动。
- [x] 建立独立实现目录。
- [x] 建立 [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md) 骨架。
- [x] 由学习者填写目标行为、实验假设和选择 SFT 的理由。
- [x] 审核数据泄漏、变量混杂、评测和算力可行性。
- [x] 审核 Gate 0A 核心计算合同，允许本机实现。
- [x] 实现并测试 `masked_sft_loss`，完成数值、梯度、单步更新、tiny overfit 与自回归复现验收。
- [x] 实现并测试单设备 FP32 `sft_optimizer_step`，完成不等有效长度 gradient accumulation、冻结参数与 gradient clipping 验收。
- [x] 实现并测试 SFT batch collator 与 assistant-only labels，完成右填充、显式 label mask、输入不可变和边界合同验收。
- [x] 将锁定 Qwen chat template/tokenizer 输出可靠转换为 `input_ids` 与 assistant-only labels，并通过样例审计和独立集成测试。
- [x] 完成结构化 batch、HF 风格 `outputs.logits` 适配和 collator 到 optimizer step 的本机纵向验收。
- [x] 完成可配置单卡训练器、原子 checkpoint、跨进程恢复、JSONL 指标、失败快照、checkpoint 保留上限和 CLI subprocess 验收。
- [x] 使用两次真实 OpenR1-Math 小样本审计证明 artifact hash 可复现。
- [x] 在 RTX 4060 Ti 8 GB 上使用真实 `Qwen3-0.6B-Base`、真实 assistant-only 样本和 BF16 完成两步训练及跨进程恢复。
- [x] 对固定 revision 的 93,733 条 OpenR1-Math 原始数据完成两次独立全量审计；train、validation、rejected 和 review 四个 artifact 的 SHA-256 全部一致。
- [x] 提供绑定 review artifact 哈希、可暂停恢复且保留逐条决定的人工复核 CLI。
- [x] 完成 pinned Open-R1/TRL 实现级对照，修正 warmup 取整，并使 checkpoint 自带 tokenizer、generation EOS 和推理态 cache 配置。
- [x] 增加正式训练前的最长样本 forward/backward/AdamW 显存探针；Gate 0B 冻结为单 GPU，禁止临时切换未验证的分布式路径。
- [ ] 人工复核 20 条 accepted 样本和每类 rejected 样本，确认过滤结果及污染候选处理符合合同。
- [ ] 按 [`REAL_SFT_EXECUTION_PLAN.md`](REAL_SFT_EXECUTION_PLAN.md) 完成 Gate 0B：工程骨架、数据审计、B0、shadow、smoke、恢复训练、正式 S1 和训练后评测。
- [ ] Gate 0B 关闭后，进入 token log-prob 的公式、shape、mask 与数值对齐实现。

## 5. 本地准入证据

截至 2026-09-04：

- 130 条默认测试和 1 条锁定 Qwen tokenizer 集成测试通过；Ruff 静态检查通过。
- 本地准入报告：[`evidence/local_readiness_report.json`](evidence/local_readiness_report.json)，状态为 `passed: true`。
- 全量数据候选为 train `62,212` 条、validation `2,000` 条、rejected `29,521` 条；最长 accepted sequence 为 `22,295` tokens，因此没有样本因 `32,768` 上限被截断或排除。
- 两次独立全量运行的 train/validation/rejected/review SHA-256 全部一致；自动可复现性已关闭，但人工内容复核仍是进入付费服务器前的独立门禁。
- 当前源码树已使用真实 `Qwen3-0.6B-Base` 和真实审计样本完成两个独立进程的 step 1 保存、step 2 恢复训练；训练集与源码树 SHA-256 均由准入报告核验。
- 本机兼容性预检使用 BF16 参数，sequence length 为 `105`，两步峰值 CUDA 显存均约 `5.73 GB`，两个训练进程各约 `110` 秒；它证明真实模型训练、checkpoint 和跨进程恢复链路可运行，不作为效果结论。
- 预检生成的两个 BF16 参数 checkpoint 合计约 `6.69 GiB`，且最终 checkpoint 已验证包含 config、generation config 和 tokenizer；正式 FP32 主参数/AdamW checkpoint 大小须由服务器 smoke 实测。服务器默认至少要求 `30 GiB` 空闲磁盘，训练配置最多保留最近 2 个 checkpoint，并在写入新 checkpoint 前预留原子临时副本空间。

服务器启动后先执行 `uv sync --frozen`，再运行 `scripts/server_preflight.py` 校验 CUDA/BF16、数据 hash、`uv.lock` hash 和磁盘。门禁返回非零退出码时不得启动 B0 或训练。

## 6. 代码结构

```text
independent_implementation/
├── README.md
├── EXPERIMENT_CONTRACT.md
├── REAL_SFT_EXECUTION_PLAN.md
├── configs/gate0b/
├── evidence/local_readiness_report.json
├── scripts/
├── src/
│   └── post_training_core/
└── tests/
```

每增加一层训练工程前，先为上一层建立可运行测试。
