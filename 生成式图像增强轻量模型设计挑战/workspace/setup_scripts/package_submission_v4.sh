#!/bin/bash
# v4 打包提交件: 100张4K输出 + JIT模型 + runner -> zip
# 用法: bash package_submission_v4.sh <ckpt> <out_zip>
set -ex
CKPT=$1
OUTZIP=${2:-/data/submit_v4.zip}
WORK=/data/submission
rm -rf "$WORK"; mkdir -p "$WORK/output_dir" "$WORK/model_dir"
cd /data/proj
/data/env/bin/python infer_v4.py --sd_path /data/weights/sd21base --ckpt "$CKPT" \
  --input_dir /data/test --output_dir "$WORK/output_dir" --tile 512 --overlap 64
/data/env/bin/python export_jit_v4.py /data/weights/sd21base "$CKPT" "$WORK/model_dir/your_model.pt"
cp /data/proj/submit/runner.py "$WORK/model_dir/runner.py"
cd "$WORK"
/data/env/bin/python -c "
import zipfile, os
out='$OUTZIP'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for r,_,fs in os.walk('.'):
        for f in fs:
            p=os.path.join(r,f); z.write(p, os.path.relpath(p,'.'))
print('zip', out)
"
echo PACKAGE_DONE
ls -lh "$OUTZIP"; echo "outputs: $(ls $WORK/output_dir|wc -l)"; ls -lh "$WORK/model_dir"
