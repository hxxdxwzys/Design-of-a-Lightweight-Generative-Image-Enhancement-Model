#!/bin/bash
# 下载 HQ 训练数据：DIV2K train HR (800 张 2K 图)
set -ex
mkdir -p /data/datasets
cd /data/datasets

if [ ! -d /data/datasets/DIV2K_HR ] || [ -z "$(ls -A /data/datasets/DIV2K_HR 2>/dev/null)" ]; then
  echo "downloading DIV2K_train_HR.zip ..."
  curl -L --retry 3 --connect-timeout 20 -o DIV2K_train_HR.zip \
    http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  /data/env/bin/python - <<'PY'
import zipfile, os
os.makedirs('/data/datasets/DIV2K_HR', exist_ok=True)
with zipfile.ZipFile('/data/datasets/DIV2K_train_HR.zip') as z:
    z.extractall('/data/datasets/DIV2K_HR')
print('extracted')
PY
  rm -f DIV2K_train_HR.zip
  # 若存在子目录(DIV2K_HR/DIV2K_train_HR/)则提升
  if [ -d /data/datasets/DIV2K_HR/DIV2K_train_HR ]; then
    mv /data/datasets/DIV2K_HR/DIV2K_train_HR/* /data/datasets/DIV2K_HR/
    rmdir /data/datasets/DIV2K_HR/DIV2K_train_HR
  fi
fi

echo DATA_DONE
echo "image count: $(ls /data/datasets/DIV2K_HR | wc -l)"
du -sh /data/datasets/DIV2K_HR
