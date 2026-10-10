#!/bin/bash
# 带 HF token 启动权重续传 (2026-09-02)
export PATH="$HOME/miniconda3/bin:$PATH"
export HF_TOKEN=$(cat "$HOME/p100/hf_token")
rm -f "$HOME/p100/logs/dl2.log"
nohup python "$HOME/p100/tools/p100_dl2.py" > "$HOME/p100/logs/dl2.log" 2>&1 &
echo "DL2_TOKEN_STARTED pid=$!"
