#!/bin/bash
# P100 1.4B w_clip=3.0 probe (2026-09-04): fresh dual-card, 2500 steps
# GPU0: p512_14b_w30 (w_clip 3.0, seed 0) / GPU1: p512_14b_w31 (w_clip 3.0, seed 1)
# Usage: bash ~/p100/tools/p100_w30_launch.sh [steps]
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

echo "=== check 1.4B weights ==="
ls "$P/weights/VOSR_1.4B_os/checkpoints/ema_model.safetensors" > /dev/null && echo CKPT_OK || { echo "1.4B weights missing"; exit 1; }

echo "=== GPU0: p512_14b_w30 (w_clip 3.0 seed 0, ${STEPS} steps, fresh) ==="
if pgrep -f "train_vosr_lora_14b[.]py.*p512_14b_w30" > /dev/null 2>&1; then
  echo "GPU0 w30 already running"
else
  CUDA_VISIBLE_DEVICES=0 nohup python "$P/tools/train_vosr_lora_14b.py" \
      --variant p512_14b_w30 --w-clip 3.0 --steps "$STEPS" "${COMMON[@]}" \
      > "$P/logs/p512_14b_w30.log" 2>&1 &
  echo "GPU0 PID: $!"
fi

echo "=== GPU1: p512_14b_w31 (w_clip 3.0 seed 1, ${STEPS} steps, fresh) ==="
if pgrep -f "train_vosr_lora_14b[.]py.*p512_14b_w31" > /dev/null 2>&1; then
  echo "GPU1 w31 already running"
else
  CUDA_VISIBLE_DEVICES=1 nohup python "$P/tools/train_vosr_lora_14b.py" \
      --variant p512_14b_w31 --w-clip 3.0 --steps "$STEPS" --seed 1 "${COMMON[@]}" \
      > "$P/logs/p512_14b_w31.log" 2>&1 &
  echo "GPU1 PID: $!"
fi

sleep 60
echo "=== 60s status ==="
tail -2 "$P/logs/p512_14b_w30.log" 2>/dev/null || echo "(w30 log not yet)"
echo "---"
tail -2 "$P/logs/p512_14b_w31.log" 2>/dev/null || echo "(w31 log not yet)"
echo LAUNCHED_W30
