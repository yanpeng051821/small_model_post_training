#!/bin/bash
# W4-2: 正式 SFT 训练
# 用 W3 smoke 验证过的配置 + AMP 混合精度修复 = 稳定全参数训练

set +e

source /data/workspace/minimind-practice/small_model_post_training/scripts/env.sh

OPENR1_REPO="/data/workspace/ai-projects/open-r1"
OPENR1_BASELINE_ROOT="/data/workspace/minimind-practice/small_model_post_training/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b"

echo "=== W4-2: 正式 SFT 训练 (LoRA) ==="
echo "模型: Qwen/Qwen3-0.6B-Base + LoRA (AMP fp16)"
echo "数据: Mixture-of-Thoughts/math, 93,733 条"
echo ""
echo "W4-1 冻结合同:"
echo "  config SHA256:  4c2cb34c"
echo "  manifest SHA256: 09518f57"
echo "  open-r1 commit:  1416fa0c"
echo "  max_seq_length:  4096"
echo "  max_steps:        2000 (~5h)"
echo "  LoRA:             r=16, alpha=32, dropout=0.05"
echo "  precision:        FP32 weights + AMP fp16 + LoRA"
echo ""

# 碎片化修复（W3 经验）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$OPENR1_REPO"

accelerate launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  src/open_r1/sft.py \
  --config "$OPENR1_BASELINE_ROOT/configs/sft_train.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/sft_train.log"

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SFT train: COMPLETED"
else
    echo "SFT train: FAILED (check log for details)"
fi
