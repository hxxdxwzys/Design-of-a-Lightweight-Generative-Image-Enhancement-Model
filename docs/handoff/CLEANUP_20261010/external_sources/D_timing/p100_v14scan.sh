#!/bin/bash
# P100 v14 配方种子扫描 (2026-09-04): 双源数据 (div2k 800 + flickr2k 2650) 复刻 v14
# v14 配方: w_clip 2.0 / patch 512 / batch 1 / lr 3e-5 / steps 4000 / save-every 500
# 用法: nohup bash p100_v14scan.sh <gpu> <seed1> <seed2> ... > ~/p100/logs/v14scan_gpu<gpu>.log 2>&1 &
# 崩溃: 记录后跳过该种子继续下一个 (seed 级容错)
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; shift

COMMON=(--patch 512 --batch 1 --lr 3e-5 --save-every 500
    --out "$P/runs"
    --data-dirs "$P/data/pngtxt_ori" "$P/data/flickr2k"
    --ckpt "$P/weights/VOSR_1.4B_os" --qwen-ae "$P/weights/Qwen-Image-vae-2d"
    --torch-cache "$P/weights/torch_cache"
    --deg-json "$P/tools/degradation_analysis.json"
    --jpeg-json "$P/tools/jpeg_blockiness.json"
    --pause-flag "$P/PAUSE.flag" --vosr-dir "$P/VOSR" --tools-dir "$P/tools")

for SEED in "$@"; do
  V="p512_14b_v14s${SEED}"
  LOGF="$P/logs/${V}.log"
  echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU seed=$SEED variant=$V start ==="
  if [ -f "$P/PAUSE.flag" ]; then echo "PAUSE.flag present, waiting 10min"; sleep 600; fi
  CUDA_VISIBLE_DEVICES=$GPU python "$P/tools/train_vosr_lora_14b.py" \
      --variant "$V" --w-clip 2.0 --steps 4000 --seed "$SEED" "${COMMON[@]}" \
      > "$LOGF" 2>&1
  RC=$?
  echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU seed=$SEED exit=$RC ==="
  tail -1 "$LOGF" | grep -q '\[done\]' && echo "SEED_DONE_OK" || echo "SEED_ABNORMAL (check $LOGF)"
  sleep 30
done
echo "=== [$(date '+%m-%d %H:%M:%S')] v14scan GPU$GPU ALL_SEEDS_FINISHED ==="
