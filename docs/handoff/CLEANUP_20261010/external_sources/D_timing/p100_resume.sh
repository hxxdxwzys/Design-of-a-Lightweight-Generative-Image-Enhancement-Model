#!/bin/bash
# P100 双卡续训: p768_w20/w15 从 s2500 无缝续到 6000 步 (2026-09-03)
# 对标 v7 (5000-6000 步) 的公平步数; 全状态续训 (opt/sched/rng 恢复)
set -e
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

echo "=== 卡0: p768_w20 续训 2500->6000 ==="
if pgrep -f "train_vosr_lora[.]py.*p768_w20" > /dev/null 2>&1; then
  echo "GPU0 已在跑, 跳过"
else
  CUDA_VISIBLE_DEVICES=0 nohup python $HOME/p100/tools/train_vosr_lora.py \
      --variant p768_w20 --w-clip 2.0 --steps 6000 --save-every 500 \
      --patch 768 --half --resume-from $HOME/p100/runs/p768_w20/lora_s2500.pt \
      > $HOME/p100/logs/p768_w20_resume.log 2>&1 &
  echo "GPU0 PID: $!"
fi

echo "=== 卡1: p768_w15 续训 2500->6000 ==="
if pgrep -f "train_vosr_lora[.]py.*p768_w15" > /dev/null 2>&1; then
  echo "GPU1 已在跑, 跳过"
else
  CUDA_VISIBLE_DEVICES=1 nohup python $HOME/p100/tools/train_vosr_lora.py \
      --variant p768_w15 --w-clip 1.5 --steps 6000 --save-every 500 \
      --patch 768 --half --resume-from $HOME/p100/runs/p768_w15/lora_s2500.pt \
      > $HOME/p100/logs/p768_w15_resume.log 2>&1 &
  echo "GPU1 PID: $!"
fi

sleep 60
echo "=== 60 秒后状态 ==="
tail -3 $HOME/p100/logs/p768_w20_resume.log 2>/dev/null || echo "(w20 日志未生成)"
echo "---"
tail -3 $HOME/p100/logs/p768_w15_resume.log 2>/dev/null || echo "(w15 日志未生成)"
echo RESUMED
