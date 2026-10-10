#!/bin/bash
# P100 M0: v14 全格式复刻 (png+jpg 双格式 div2k 800对 + flickr 2650 = 4250 图)
# v14 配方: w_clip 2.0 / patch512 / batch1 / lr3e-5 / save-every 500
# 本实验: 6000 步 (复刻判定区 s2400-4000 + 长训世代效应 s5000-6000)
# 用法: nohup bash p100_m0_scan.sh <gpu> <seed> > ~/p100/logs/m0_scan_gpu<gpu>.log 2>&1 &
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; SEED=$2
V="p512_14b_m0s${SEED}"
LOGF="$P/logs/${V}.log"

echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU seed=$SEED variant=$V start ==="
if [ -f "$P/PAUSE.flag" ]; then echo "PAUSE.flag present, waiting 10min"; sleep 600; fi
CUDA_VISIBLE_DEVICES=$GPU python "$P/tools/train_vosr_lora_14b.py" \
    --variant "$V" --w-clip 2.0 --steps 6000 --seed "$SEED" \
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
echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU seed=$SEED exit=$RC ==="
tail -1 "$LOGF" | grep -q '\[done\]' && echo "SEED_DONE_OK" || echo "SEED_ABNORMAL"
echo "=== M0 GPU$GPU FINISHED ==="
