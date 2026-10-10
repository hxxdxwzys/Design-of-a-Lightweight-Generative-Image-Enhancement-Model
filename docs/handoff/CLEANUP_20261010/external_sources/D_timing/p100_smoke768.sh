#!/bin/bash
# P100 patch768 显存冒烟 (2026-09-02): 测 OOM 与步速
export PATH="$HOME/miniconda3/bin:$PATH"
export VOSR_DIR=$HOME/p100/VOSR
export VOSR_CKPT=$HOME/p100/weights/VOSR_0.5B_os
export VOSR_AE=$HOME/p100/weights/sd21base
export VOSR_LWDEC=$HOME/p100/weights/sd21_lwdecoder.pth
export VOSR_CACHE=$HOME/p100/weights/torch_cache
export VOSR_DATA="$HOME/p100/data/flickr2k"
export VOSR_OUT=$HOME/p100/runs
export EMP_DEG=$HOME/p100/tools/degradation_analysis.json
export EMP_JPEG=$HOME/p100/tools/jpeg_blockiness.json
export PAUSE_FLAG=$HOME/p100/PAUSE.flag
CUDA_VISIBLE_DEVICES=0 python $HOME/p100/tools/train_vosr_lora.py \
    --variant smoke_p768 --w-clip 2.0 --steps 3 --save-every 9999 \
    --patch 768 --half > $HOME/p100/logs/smoke_p768.log 2>&1
echo "SMOKE_EXIT=$?"
tail -6 $HOME/p100/logs/smoke_p768.log
