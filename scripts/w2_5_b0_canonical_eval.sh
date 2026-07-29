#!/bin/bash
# W2-5: B0 Canonical Eval — 完整 MATH-500 评测
# 目的: 跑出 B0 baseline 分数，冻结合同后 B1/B2 必须复用同样参数
#
# 评测合同（冻结后不再更改）:
#   - 模型: Qwen/Qwen3-0.6B-Base (float16)
#   - temperature: 0.6 (采样, 允许 4 个答案/题估算 pass@1)
#   - max_new_tokens: 512
#   - max_model_length: 4096
#   - chat_template: Qwen3 默认 (含 <think>)
#   - gpu_memory_utilization: 0.80
#   - MATH-500 全部 500 题

set -e

: ${OPENR1_BASELINE_ROOT:?请先 source scripts/env.sh}

echo "=== W2-5: B0 Canonical Eval (完整 MATH-500) ==="
echo "模型: Qwen/Qwen3-0.6B-Base (float16)"
echo "评测集: MATH-500 (全部 500 题)"
echo "参数: temperature=0.6, max_new_tokens=512, 4 answers/question"
echo "预计时间: 30-60分钟"
echo ""

export VLLM_WORKER_MULTIPROC_METHOD=spawn

B0_MODEL="Qwen/Qwen3-0.6B-Base"
B0_MODEL_ARGS="model_name=$B0_MODEL,dtype=float16,max_model_length=4096,max_num_batched_tokens=4096,gpu_memory_utilization=0.80,generation_parameters={max_new_tokens:512,temperature:0.6}"

lighteval vllm "$B0_MODEL_ARGS" "lighteval|math_500|0|0" \
  --use-chat-template \
  --save-details \
  --output-dir "$OPENR1_BASELINE_ROOT/evals/b0-canonical" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/b0_lighteval_canonical.log"
