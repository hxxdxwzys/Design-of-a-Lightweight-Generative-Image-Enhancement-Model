#!/bin/bash
# 用 checkpoint 对验证集 LQ 推理并评测。用法: bash eval_ckpt.sh <ckpt_path>
CKPT=${1:-/data/proj/runs/v1/unet_step1000.pth}
cd /data/proj
echo "=== infer $CKPT ==="
rm -rf /data/val/enhanced
/data/env/bin/python infer.py --sd_path /data/weights/sd21base --ckpt "$CKPT" \
  --input_dir /data/val/lq --output_dir /data/val/enhanced --tile 512 --overlap 64 2>&1 | tail -6
echo "=== eval ==="
/data/env/bin/python eval.py --pred-dir /data/val/enhanced --gt-dir /data/val/gt \
  --gt-pattern '*' --out /data/val/eval_$(basename "$CKPT" .pth).json 2>&1 \
  | grep -v -E "%\||Downloading|kB/s|MB/s|Loading" | tail -20
