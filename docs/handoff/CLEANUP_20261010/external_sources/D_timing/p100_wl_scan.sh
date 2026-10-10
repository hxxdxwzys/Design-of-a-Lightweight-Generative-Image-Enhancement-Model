#!/bin/bash
# P100 w_lpips 轴 (2026-09-09): m0 冠军配方 (w_clip 2.0 / 6000 步 / 双源 4250) 仅改 w_lpips
# 假说 (BRAINSTORM §2.2): 更强保真锚 -> 同 clip 下 lpips 更低 = Pareto 前沿下移
# 未探索区: w_lpips/w_clip 比值 1.0-1.5 (已扫 w_clip 1.5-3.0 @ w_lpips 1.0 = 比值 0.33-0.67)
# 用法: nohup bash p100_wl_scan.sh <gpu> <w_lpips:整数> <seed> > ~/p100/logs/wlscan_gpu<gpu>.log 2>&1 &
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
GPU=$1; WL=$2; SEED=$3
V="p512_14b_m0s${SEED}wl${WL}"
LOGF="$P/logs/${V}.log"

echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU w_lpips=$WL seed=$SEED variant=$V start ==="
if [ -f "$P/PAUSE.flag" ]; then echo "PAUSE.flag present, waiting 10min"; sleep 600; fi
CUDA_VISIBLE_DEVICES=$GPU python "$P/tools/train_vosr_lora_14b.py" \
    --variant "$V" --w-clip 2.0 --w-lpips "$WL" --steps 6000 --seed "$SEED" \
    --rank 32 --alpha 64 \
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
echo "=== [$(date '+%m-%d %H:%M:%S')] GPU$GPU w_lpips=$WL exit=$RC ==="
tail -1 "$LOGF" | grep -q '\[done\]' && echo "WL_DONE_OK" || echo "WL_ABNORMAL"
