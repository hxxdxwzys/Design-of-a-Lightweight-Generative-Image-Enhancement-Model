#!/bin/bash
# P100 1.4B patch512 冒烟 (2026-09-03): 测 OOM 与步速 (GPU0) — 显式传参版
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"
CUDA_VISIBLE_DEVICES=0 python "$P/tools/train_vosr_lora_14b.py" \
    --variant smoke_14b_p512 --w-clip 2.0 --steps 3 --save-every 99999 \
    --patch 512 --batch 1 --lr 3e-5 \
    --out "$P/runs" \
    --data-dirs "$P/data/flickr2k" \
    --ckpt "$P/weights/VOSR_1.4B_os" \
    --qwen-ae "$P/weights/Qwen-Image-vae-2d" \
    --torch-cache "$P/weights/torch_cache" \
    --deg-json "$P/tools/degradation_analysis.json" \
    --jpeg-json "$P/tools/jpeg_blockiness.json" \
    --pause-flag "$P/PAUSE.flag" \
    --vosr-dir "$P/VOSR" \
    --tools-dir "$P/tools" \
    > "$P/logs/smoke_14b.log" 2>&1
echo "SMOKE_EXIT=$?"
tail -5 "$P/logs/smoke_14b.log"
