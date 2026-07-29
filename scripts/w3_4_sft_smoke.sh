#!/bin/bash
# W3-4: SFT Smoke Test
# 用 sft.py 跑最小训练闭环：2-4 个 optimizer step

set +e  # 不遇错退出，需要完整日志

source /data/workspace/minimind-practice/small_model_post_training/scripts/env.sh

OPENR1_REPO="/data/workspace/ai-projects/open-r1"
OPENR1_BASELINE_ROOT="/data/workspace/minimind-practice/small_model_post_training/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b"

echo "=== W3-4: SFT Smoke Test ==="
echo "模型: Qwen/Qwen3-0.6B-Base → SFT"
echo "数据: Mixture-of-Thoughts/math (shuffle seed=42)"
echo "Smoke: max_steps=4, batch=1, grad_accum=4"
echo ""

# 单个 T4，不需要 multi-GPU accelerator config
# accelerate 默认用单 GPU

cd "$OPENR1_REPO"

# 防止 OOM 碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  src/open_r1/sft.py \
  --config "$OPENR1_BASELINE_ROOT/configs/sft_smoke.yaml" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/sft_smoke.log"

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SFT smoke: PASSED"
else
    echo "SFT smoke: FAILED (check log for details)"
fi
