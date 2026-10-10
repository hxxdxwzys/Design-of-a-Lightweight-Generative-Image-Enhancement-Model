#!/bin/bash
# P100 1.4B 双卡训练启动 v2 (2026-09-03): 显式传参版
# GPU0: p512_14b_w20 (patch512, w_clip 2.0) / GPU1: p512_14b_w15 (patch512, w_clip 1.5)
# 用法: bash p100_launch_14b.sh [steps]   (默认 2500)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
STEPS=${1:-2500}

COMMON=(--patch 512 --batch 1 --lr 3e-5 --save-every 500
    --out "$P/runs" --data-dirs "$P/data/flickr2k"
    --ckpt "$P/weights/VOSR_1.4B_os" --qwen-ae "$P/weights/Qwen-Image-vae-2d"
    --torch-cache "$P/weights/torch_cache"
    --deg-json "$P/tools/degradation_analysis.json"
    --jpeg-json "$P/tools/jpeg_blockiness.json"
    --pause-flag "$P/PAUSE.flag" --vosr-dir "$P/VOSR" --tools-dir "$P/tools")

echo "=== 检查 1.4B 权重 ==="
ls "$P/weights/VOSR_1.4B_os/checkpoints/ema_model.safetensors" > /dev/null && echo CKPT_OK || { echo "1.4B 权重缺失"; exit 1; }

echo "=== 卡0: p512_14b_w20 (${STEPS} 步) ==="
if pgrep -f "train_vosr_lora_14b[.]py.*p512_14b_w20" > /dev/null 2>&1; then
  echo "GPU0 已在跑"
else
  CUDA_VISIBLE_DEVICES=0 nohup python "$P/tools/train_vosr_lora_14b.py" \
      --variant p512_14b_w20 --w-clip 2.0 --steps "$STEPS" "${COMMON[@]}" \
      > "$P/logs/p512_14b_w20.log" 2>&1 &
  echo "GPU0 PID: $!"
fi

echo "=== 卡1: p512_14b_w15 (${STEPS} 步) ==="
if pgrep -f "train_vosr_lora_14b[.]py.*p512_14b_w15" > /dev/null 2>&1; then
  echo "GPU1 已在跑"
else
  CUDA_VISIBLE_DEVICES=1 nohup python "$P/tools/train_vosr_lora_14b.py" \
      --variant p512_14b_w15 --w-clip 1.5 --steps "$STEPS" "${COMMON[@]}" \
      > "$P/logs/p512_14b_w15.log" 2>&1 &
  echo "GPU1 PID: $!"
fi

sleep 60
echo "=== 60 秒后状态 ==="
tail -3 "$P/logs/p512_14b_w20.log" 2>/dev/null || echo "(w20 日志未生成)"
echo "---"
tail -3 "$P/logs/p512_14b_w15.log" 2>/dev/null || echo "(w15 日志未生成)"
echo LAUNCHED_14B
