# Open-R1 Baseline Checklist

本清单只关闭项目级 L0 Open-R1 八周运行子路线；后续课程、CS336 终验和论文复现 gate 见[项目级主路线](../small_model_post_training_research_and_roadmap.md)。

## Identity

- baseline id：`open-r1-qwen3-0.6b`
- route：`reproduce`
- owner stage：`baseline`
- current phase：`b1_evidence_and_eval_sync`
- trust status：`b0_reported_evidence_pointer_missing`

## Core

- [x] Baseline 对象和 reproduce 路线已经明确。
- [x] 数据、split 和 metric contract 已明确到足以判断可比性；硬件相关数值有正式运行前测量门。
- [x] `PLAN.md` 已记录已知命令入口、预期输出、验收条件和 fallback。
- [x] Source commit/version 已固定：`1416fa0cf21595d2083b399a2a0bbddd7f6e9563`。
- [x] Smoke decision 已完成并有持久证据。（W2-4 smoke: vLLM + LightEval 链路通过）
- [x] 正式 validation/run 决策已明确。（W2-5 canonical: 500题, temperature=0.6, 1n+4n pass@1）
- [x] 预期结果文件和要求指标已经核对并写入 `PLAN.md`。
- [ ] Baseline 已以 accepted、blocked 或 waived 之一关闭。

## Weekly Gates

- [x] Week 1：完成源码地图、版本固定和复现合同。
- [ ] Week 2：执行报告称 B0 MATH-500 ≈ 27%；需补结果摘要、实际命令和服务器证据指针后关闭。
- [ ] Week 3：数据与 collator 审计完成；历史 SFT smoke 第 2-4 step 出现 `loss=0`、`grad_norm=NaN`，需补与后续 B1 正式训练的关系和根因证据。
- [ ] Week 4：B1 已报告训练完成；补齐最终配置、实际命令、日志、checkpoint 指针/hash 和可用性测试后关闭。
- [ ] Week 5：B1 LightEval 进行中或结果待同步；补齐 B0/B1 同合同对照与失败分析后关闭。
- [ ] Week 6：完成 GRPO smoke test。
- [ ] Week 7：完成 GRPO 缩放复现。
- [ ] Week 8：完成 B0/B1/B2 同条件验证与 baseline 判定。

## Closeout

- [ ] 写出 1-2 句 baseline 信任结论。
- [ ] 明确下一阶段是 TinyTutor 迁移、repair 还是停止。

## Optional Expansion

- [x] Source paper 已识别。
- [x] Source repo 已识别。
- [x] Paper 已读到足以准确复述本次涉及的 distillation、R1-Zero 与 multi-stage 方法边界。
- [x] Repo 已读到足以确认 SFT、GRPO、reward 与单 GPU evaluation entrypoints。
- [x] 工作目录和服务器路径已确认。
- [x] `uv` 环境路线已验证。（LD_PRELOAD fix applied）
- [x] 关键依赖已核对。
- [x] 模型与数据下载路径已确认。
- [ ] 所有执行偏差已同步回 `PLAN.md`。
- [ ] `verification.md` 已完成。
