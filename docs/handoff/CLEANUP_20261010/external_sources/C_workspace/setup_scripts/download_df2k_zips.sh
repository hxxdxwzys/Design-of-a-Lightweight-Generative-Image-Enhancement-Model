#!/bin/bash
# DF2K 下载 (上海, hf-mirror, 用 yangtao9009 的 zip 而非 bdanko parquet 避免 xet)
set -ex
TOKEN=$(cat /data/hf_token)
mkdir -p /data/datasets/DF2K

# 1) DIV2K_train.zip (3.5G, 普通 CDN)
if [ ! -f /data/datasets/.div2k_done ]; then
  curl -L -H "Authorization: Bearer $TOKEN" -o /data/datasets/DIV2K_train.zip \
    "https://hf-mirror.com/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip"
  /data/env/bin/python - <<'PY'
import zipfile, os, shutil
out='/data/datasets/DF2K'; tmp='/data/datasets/DIV2K_raw'
os.makedirs(tmp, exist_ok=True)
with zipfile.ZipFile('/data/datasets/DIV2K_train.zip') as z:
    z.extractall(tmp)
n=0
for root,_,files in os.walk(tmp):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
            shutil.move(os.path.join(root,f), os.path.join(out,'div2k_%04d_%s'%(n,f))); n+=1
shutil.rmtree(tmp, ignore_errors=True)
print('DIV2K done', n, flush=True)
PY
  rm -f /data/datasets/DIV2K_train.zip
  touch /data/datasets/.div2k_done
fi

# 2) Flickr2K.zip (11G)
if [ ! -f /data/datasets/.flickr2k_done ]; then
  curl -L -H "Authorization: Bearer $TOKEN" -o /data/datasets/Flickr2K.zip \
    "https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"
  /data/env/bin/python - <<'PY'
import zipfile, os, shutil
out='/data/datasets/DF2K'; tmp='/data/datasets/Flickr2K_raw'
os.makedirs(tmp, exist_ok=True)
with zipfile.ZipFile('/data/datasets/Flickr2K.zip') as z:
    z.extractall(tmp)
n=0
for root,_,files in os.walk(tmp):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.webp')):
            shutil.move(os.path.join(root,f), os.path.join(out,'flickr_%04d_%s'%(n,f))); n+=1
shutil.rmtree(tmp, ignore_errors=True)
print('Flickr2K done', n, flush=True)
PY
  rm -f /data/datasets/Flickr2K.zip
  touch /data/datasets/.flickr2k_done
fi

echo DF2K_ZIP_DONE
echo "DF2K count: $(ls /data/datasets/DF2K|wc -l)"
du -sh /data/datasets/DF2K
