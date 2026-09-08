#!/bin/bash
# W4-2: 正式 SFT 训练
# 使用 LoRA + AMP 的 W4 候选配置；必须先通过 smoke gate

set +e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CONFIG="$OPENR1_BASELINE_ROOT/configs/sft_train.yaml"
MANIFEST="$OPENR1_BASELINE_ROOT/data/sft_manifest.json"
CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)
MANIFEST_SHA256=$(sha256sum "$MANIFEST" | cut -d' ' -f1)

echo "=== W4-2: 正式 SFT 训练 (LoRA) ==="
echo "模型: Qwen/Qwen3-0.6B-Base + LoRA (AMP fp16)"
echo "数据: Mixture-of-Thoughts/math, 93,733 条"
echo ""
echo "W4-1 冻结合同:"
echo "  config SHA256:  $CONFIG_SHA256"
echo "  manifest SHA256: $MANIFEST_SHA256"
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
mkdir -p "$OPENR1_BASELINE_ROOT/logs"

accelerate launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  src/open_r1/sft.py \
  --config "$CONFIG" \
  --output_dir "$OPENR1_BASELINE_ROOT/checkpoints/sft_train" \
  2>&1 | tee "$OPENR1_BASELINE_ROOT/logs/sft_train.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SFT train: COMPLETED"
else
    echo "SFT train: FAILED (check log for details)"
fi
