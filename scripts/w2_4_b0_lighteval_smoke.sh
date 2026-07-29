#!/bin/bash
# W2-4: B0 LightEval MATH-500 五题 smoke
# 目的: 验证 vLLM + LightEval 评测链路在 T4 上可用
# 只跑 5 题，结果不作为 baseline（不允许从 5 题推 500 题）

set -e

# 环境变量（依赖 source scripts/env.sh）
: ${OPENR1_BASELINE_ROOT:?请先 source scripts/env.sh}

echo "=== W2-4: B0 LightEval 五题 smoke ==="
echo "模型: Qwen/Qwen3-0.6B-Base (float16)"
echo "评测集: MATH-500 (仅 5 题)"
echo ""

# LightEval 参数说明:
#   lighteval|math_500|0|0             → task=math_500, 取 0 个子集
#   --max-samples 5                     → 只跑 5 题（smoke test）
#   --use-chat-template                 → 用 tokenizer 的 chat template
#   --save-details                      → 保存每题详细结果
#   temperature=0.0                     → 确定性输出
#   gpu_memory_utilization=0.80        → 给 vLLM 80% 可用显存

export VLLM_WORKER_MULTIPROC_METHOD=spawn

B0_MODEL="Qwen/Qwen3-0.6B-Base"
B0_MODEL_ARGS="model_name=$B0_MODEL,dtype=float16,max_model_length=4096,max_num_batched_tokens=4096,gpu_memory_utilization=0.80,generation_parameters={max_new_tokens:512,temperature:0.6}"

lighteval vllm "$B0_MODEL_ARGS" "lighteval|math_500|0|0" \
  --use-chat-template \
  --max-samples 5 \
  --save-details \
  --output-dir "$OPENR1_BASELINE_ROOT/evals/b0-smoke" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/b0_lighteval_smoke.log"
