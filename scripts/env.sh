#!/bin/bash
# Week 2 统一环境变量
# 用法: source scripts/env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POST_TRAINING_ROOT="${POST_TRAINING_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# fix flash_attn GLIBCXX 3.4.29 不兼容
export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6

# 使用 open-r1 安装的 python 环境
export PATH="/data/workspace/ai-projects/openr1-venv/bin:$PATH"

# 仓库位置可变；显式环境变量优先，否则根据当前脚本位置推导。
export OPENR1_REPO="${OPENR1_REPO:-/data/workspace/ai-projects/open-r1}"
export OPENR1_BASELINE_ROOT="${OPENR1_BASELINE_ROOT:-$POST_TRAINING_ROOT/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b}"

echo "[env] openr1-venv python: $(which python)"
echo "[env] baseline root: $OPENR1_BASELINE_ROOT"
