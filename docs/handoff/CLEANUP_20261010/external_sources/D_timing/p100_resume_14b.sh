#!/bin/bash
# P100 1.4B 续训 2500->6000 (2026-09-04): 全状态无缝续训, 双卡
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
TARGET=6000

COMMON=(--patch 512 --batch 1 --lr 3e-5 --save-every 500
    --out "$P/runs" --data-dirs "$P/data/flickr2k"
    --ckpt "$P/weights/VOSR_1.4B_os" --qwen-ae "$P/weights/Qwen-Image-vae-2d"
    --torch-cache "$P/weights/torch_cache"
    --deg-json "$P/tools/degradation_analysis.json"
    --jpeg-json "$P/tools/jpeg_blockiness.json"
    --pause-flag "$P/PAUSE.flag" --vosr-dir "$P/VOSR" --tools-dir "$P/tools")

echo "=== 卡0: p512_14b_w20 续训 -> $TARGET ==="
CUDA_VISIBLE_DEVICES=0 nohup python "$P/tools/train_vosr_lora_14b.py" \
    --variant p512_14b_w20 --w-clip 2.0 --steps "$TARGET" "${COMMON[@]}" \
    --resume-from "$P/runs/p512_14b_w20/lora_s2500.pt" \
    > "$P/logs/p512_14b_w20_resume.log" 2>&1 &
echo "GPU0 PID: $!"

echo "=== 卡1: p512_14b_w15 续训 -> $TARGET ==="
CUDA_VISIBLE_DEVICES=1 nohup python "$P/tools/train_vosr_lora_14b.py" \
    --variant p512_14b_w15 --w-clip 1.5 --steps "$TARGET" "${COMMON[@]}" \
    --resume-from "$P/runs/p512_14b_w15/lora_s2500.pt" \
    > "$P/logs/p512_14b_w15_resume.log" 2>&1 &
echo "GPU1 PID: $!"

sleep 60
echo "=== 60 秒后 ==="
grep -E 'resume|boot' "$P/logs/p512_14b_w20_resume.log" | tail -3
echo "---"
grep -E 'resume|boot' "$P/logs/p512_14b_w15_resume.log" | tail -3
echo RESUMED_14B
