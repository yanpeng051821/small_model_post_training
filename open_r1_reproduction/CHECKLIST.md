# Open-R1 Baseline Checklist

## Identity

- baseline id：`open-r1-qwen3-0.6b`
- route：`reproduce`
- owner stage：`baseline`
- current phase：`setup`
- trust status：`unverified`

## Core

- [x] Baseline 对象和 reproduce 路线已经明确。
- [x] 数据、split 和 metric contract 已明确到足以判断可比性；硬件相关数值有正式运行前测量门。
- [x] `PLAN.md` 已记录已知命令入口、预期输出、验收条件和 fallback。
- [x] Source commit/version 已固定：`1416fa0cf21595d2083b399a2a0bbddd7f6e9563`。
- [ ] Smoke decision 已完成并有持久证据。
- [ ] 正式 validation/run 决策已明确。
- [x] 预期结果文件和要求指标已经核对并写入 `PLAN.md`。
- [ ] Baseline 已以 accepted、blocked 或 waived 之一关闭。

## Weekly Gates

- [x] Week 1：完成源码地图、版本固定和复现合同。
- [ ] Week 2：完成环境与 B0 baseline。
- [ ] Week 3：完成数据审计和 SFT smoke test。
- [ ] Week 4：完成 SFT 缩放复现。
- [ ] Week 5：完成 B1 验证与失败分析。
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
- [ ] 工作目录和服务器路径已确认。
- [ ] `uv` 环境路线已验证。
- [ ] 关键依赖已核对。
- [ ] 模型与数据下载路径已确认。
- [ ] 所有执行偏差已同步回 `PLAN.md`。
- [ ] `verification.md` 已完成。
