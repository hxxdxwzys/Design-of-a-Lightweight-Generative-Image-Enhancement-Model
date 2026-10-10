#!/bin/bash
# 打包提交件: 100张4K输出 + JIT模型 + runner -> zip
# 用法: bash package_submission.sh <ckpt> <out_zip>
set -ex
CKPT=$1
OUTZIP=${2:-/data/submit.zip}
WORK=/data/submission
rm -rf "$WORK"
mkdir -p "$WORK/output_dir" "$WORK/model_dir"

cd /data/proj

# 1) 生成 100 张 4K 输出
/data/env/bin/python infer.py --sd_path /data/weights/sd21base --ckpt "$CKPT" \
  --input_dir /data/test --output_dir "$WORK/output_dir" --tile 512 --overlap 64

# 2) 导出 JIT (fp16)
/data/env/bin/python export_jit.py /data/weights/sd21base "$CKPT" "$WORK/model_dir/your_model.pt"

# 3) runner.py
cp /data/proj/submit/runner.py "$WORK/model_dir/runner.py"

# 4) 打包 zip (Python, 避免 zip 命令缺失)
cd "$WORK"
/data/env/bin/python - <<PY
import zipfile, os
out = "$OUTZIP"
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk('.'):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, '.'))
print('zip created', out)
PY
echo "PACKAGE_DONE"
ls -lh "$OUTZIP"
echo "output count: $(ls $WORK/output_dir | wc -l)"
echo "model_dir:"; ls -lh "$WORK/model_dir"
