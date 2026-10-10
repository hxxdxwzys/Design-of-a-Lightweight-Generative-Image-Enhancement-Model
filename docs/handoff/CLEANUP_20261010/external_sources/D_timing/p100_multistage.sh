#!/bin/bash
# 多阶段探针 (2026-09-07): 从 m0s1-s4000 resume, w_clip 2.0→1.0 续训至 6000
# 假设: clip 已推到 0.934 高位后, 弱化 clip 推力可让 lpips 回落而 clip 守住
# 注意: --steps 6000 与原始一致 (T_max 不 mismatch); 存档含 sched_state 自动续
set -u
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
V="p512_14b_m0s1w1"
LOGF="$P/logs/${V}.log"
echo "=== [$(date '+%m-%d %H:%M:%S')] multistage probe start ==="
CUDA_VISIBLE_DEVICES=0 python "$P/tools/train_vosr_lora_14b.py" \
    --variant "$V" --w-clip 1.0 --steps 6000 --save-every 500 \
    --patch 512 --batch 1 --lr 3e-5 \
    --out "$P/runs" \
    --data-dirs "$P/data/pngtxt_ori" "$P/data/flickr2k" \
    --ckpt "$P/weights/VOSR_1.4B_os" --qwen-ae "$P/weights/Qwen-Image-vae-2d" \
    --torch-cache "$P/weights/torch_cache" \
    --deg-json "$P/tools/degradation_analysis.json" \
    --jpeg-json "$P/tools/jpeg_blockiness.json" \
    --pause-flag "$P/PAUSE.flag" --vosr-dir "$P/VOSR" --tools-dir "$P/tools" \
    --resume-from "$P/runs/p512_14b_m0s1/lora_s4000.pt" \
    > "$LOGF" 2>&1
RC=$?
echo "=== exit=$RC ==="
tail -1 "$LOGF" | grep -q '\[done\]' && echo "DONE_OK" || echo "ABNORMAL"
echo "=== MULTISTAGE FINISHED ==="
