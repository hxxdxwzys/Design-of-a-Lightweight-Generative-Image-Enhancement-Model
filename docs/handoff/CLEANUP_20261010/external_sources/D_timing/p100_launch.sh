#!/bin/bash
# P100 双卡训练启动 v2 (2026-09-02): GPU0 p768_w20 / GPU1 p768_w15 (patch768, 16GB 实测可行)
# 用法: bash p100_launch.sh; 查看日志: tail -f ~/p100/logs/p768_*.log
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

echo "=== 数据/权重检查 ==="
ls "$VOSR_CKPT/checkpoints/" | head -3
ls -d "$VOSR_CACHE/facebookresearch_dinov2_main" && echo DINOV2_OK
N=$(find "$VOSR_DATA" -name '*.png' | wc -l); echo "训练图 $N 张"
[ "$N" -gt 2000 ] || { echo "数据不足, 中止"; exit 1; }

echo "=== 卡0: p768_w20 (patch768, w_clip 2.0, 2500 步) ==="
if pgrep -f "train_vosr_lora[.]py.*p768_w20" > /dev/null 2>&1; then
  echo "GPU0 已在跑, 跳过"
else
  CUDA_VISIBLE_DEVICES=0 nohup python $HOME/p100/tools/train_vosr_lora.py \
      --variant p768_w20 --w-clip 2.0 --steps 2500 --save-every 500 \
      --patch 768 --half > $HOME/p100/logs/p768_w20.log 2>&1 &
  echo "GPU0 PID: $!"
fi

echo "=== 卡1: p768_w15 (patch768, w_clip 1.5, 2500 步) ==="
if pgrep -f "train_vosr_lora[.]py.*p768_w15" > /dev/null 2>&1; then
  echo "GPU1 已在跑, 跳过"
else
  CUDA_VISIBLE_DEVICES=1 nohup python $HOME/p100/tools/train_vosr_lora.py \
      --variant p768_w15 --w-clip 1.5 --steps 2500 --save-every 500 \
      --patch 768 --half > $HOME/p100/logs/p768_w15.log 2>&1 &
  echo "GPU1 PID: $!"
fi

sleep 45
echo "=== 45 秒后状态 ==="
tail -4 $HOME/p100/logs/p768_w20.log 2>/dev/null || echo "(w20 日志未生成)"
echo "---"
tail -4 $HOME/p100/logs/p768_w15.log 2>/dev/null || echo "(w15 日志未生成)"
echo LAUNCHED
