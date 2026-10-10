#!/bin/bash
# P100 M0 种子补扫 (2026-09-05): m0s1-s4, 同 m0 配置 (4250图 w2.0/6000步)
# 用法: nohup bash p100_m0_scan2.sh <gpu> <seed1> <seed2> > ~/p100/logs/m0scan2_gpu<gpu>.log 2>&1 &
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; shift

for SEED in "$@"; do
  V="p512_14b_m0s${SEED}"
  LOGF="$P/logs/${V}.log"
  echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU seed=$SEED start ==="
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
  sleep 30
done
echo "=== M0-scan2 GPU$GPU ALL_FINISHED ==="
