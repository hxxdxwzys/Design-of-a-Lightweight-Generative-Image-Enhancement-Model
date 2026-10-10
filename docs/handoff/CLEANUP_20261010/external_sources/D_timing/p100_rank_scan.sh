#!/bin/bash
# P100 rank 收缩轴 (2026-09-09): m0 冠军配方 (w_clip 2.0 / 6000 步 / 双源 4250) 仅改 LoRA rank
# 假说 (BRAINSTORM §2.1): 小 rank 限制自由度 -> 更温和的 clip 推 -> 突破 lpips 地板带 0.357
# 用法: nohup bash p100_rank_scan.sh <gpu> <rank> <seed> > ~/p100/logs/rankscan_gpu<gpu>.log 2>&1 &
#   alpha = 2*rank (保持 alpha/r = 2, 与冠军 r32/alpha64 同比例)
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; RANK=$2; SEED=$3
ALPHA=$((RANK * 2))
V="p512_14b_m0s${SEED}r${RANK}"
LOGF="$P/logs/${V}.log"

echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU rank=$RANK alpha=$ALPHA seed=$SEED start ==="
if [ -f "$P/PAUSE.flag" ]; then echo "PAUSE.flag present, waiting 10min"; sleep 600; fi
CUDA_VISIBLE_DEVICES=$GPU python "$P/tools/train_vosr_lora_14b.py" \
    --variant "$V" --w-clip 2.0 --steps 6000 --seed "$SEED" \
    --rank "$RANK" --alpha "$ALPHA" \
    --patch 512 --batch 1 --lr 3e-5 --save-every 500 \
    --out "$P/runs" \
    --data-dirs "$P/data/pngtxt_ori" "$P/data/flickr2k" \
    --ckpt "$P/weights/VOSR_1.4B_os" --qwen-ae "$P/weights/Qwen-Image-vae-2d" \
    --torch-cache "$P/weights/torch_cache" \
    --deg-json "$P/tools/degradation_analysis.json" \
    --jpeg-json "$P/tools/jpeg_blockiness.json" \
    --pause-flag "$P/PAUSE.flag" --vosr-dir "$P/VOSR" --tools-dir "$P/tools" \
    > "$LOGF" 2>&1
RC=$?
echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU rank=$RANK exit=$RC ==="
tail -1 "$LOGF" | grep -q '\[done\]' && echo "RANK_DONE_OK" || echo "RANK_ABNORMAL"
