#!/bin/bash
# P100 世代3 (2026-09-07): 经验继承种子 {1,7}
#   GPU0 线A: w_clip 1.8 / T6000 x 种子 {1,7}  (弱感知 frontier 下移)
#   GPU1 线B: w_clip 2.0 / T8000 x 种子 {1,7}  (长训平台拓宽)
# 用法: nohup bash p100_gen3.sh <gpu> <wclip> <steps> <seed1> <seed2> > ~/p100/logs/gen3_gpu<gpu>.log 2>&1 &
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; WCLIP=$2; STEPS=$3; shift 3

for SEED in "$@"; do
  V="p512_14b_g3w${WCLIP/./}s${SEED}"
  LOGF="$P/logs/${V}.log"
  echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU w=$WCLIP steps=$STEPS seed=$SEED variant=$V start ==="
  if [ -f "$P/PAUSE.flag" ]; then echo "PAUSE.flag present, waiting 10min"; sleep 600; fi
  CUDA_VISIBLE_DEVICES=$GPU python "$P/tools/train_vosr_lora_14b.py" \
      --variant "$V" --w-clip "$WCLIP" --steps "$STEPS" --seed "$SEED" \
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
echo "=== GEN3 GPU$GPU ALL_FINISHED ==="
