#!/bin/bash
# Week 2 统一环境变量
# 用法: source scripts/env.sh

# fix flash_attn GLIBCXX 3.4.29 不兼容
export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6

# 使用 open-r1 安装的 python 环境
export PATH="/data/workspace/ai-projects/openr1-venv/bin:$PATH"

# baseline 输出根目录
export OPENR1_BASELINE_ROOT="/data/workspace/minimind-practice/small_model_post_training/open_r1_reproduction/baselines/local/open-r1-qwen3-0.6b"

echo "[env] openr1-venv python: $(which python)"
echo "[env] baseline root: $OPENR1_BASELINE_ROOT"
