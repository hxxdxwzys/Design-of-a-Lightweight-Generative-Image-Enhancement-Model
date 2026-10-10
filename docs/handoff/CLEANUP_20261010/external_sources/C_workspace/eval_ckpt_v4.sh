#!/bin/bash
# v4 评测: infer_v4 val LQ + eval vs GT。用法: bash eval_ckpt_v4.sh <ckpt>
CKPT=${1:-/data/proj/runs/v4/v4_final.pth}
cd /data/proj
echo "=== infer $CKPT ==="
rm -rf /data/val/enhanced
/data/env/bin/python infer_v4.py --sd_path /data/weights/sd21base --ckpt "$CKPT" \
  --input_dir /data/val/lq --output_dir /data/val/enhanced --tile 512 --overlap 64 2>&1 | tail -5
echo "=== eval ==="
/data/env/bin/python eval.py --pred-dir /data/val/enhanced --gt-dir /data/val/gt \
  --gt-pattern '*' --out /data/val/eval_v4_$(basename "$CKPT" .pth).json 2>&1 \
  | grep -v -E "%\||Downloading|kB/s|MB/s|Loading|FutureWarning|warnings.warn" | tail -16
