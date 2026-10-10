#!/bin/bash
# 提取 DF2K = DIV2K(800) + Flickr2K(2650) -> /data/datasets/DF2K/  (v2 训练数据)
# 需在 v1 训练结束后运行(CPU 空闲时才快)
set -ex
export HF_TOKEN=$(cat /data/hf_token)
mkdir -p /data/datasets/DF2K

# 1) DIV2K 全量 800
if [ ! -f /data/datasets/.div2k_done ]; then
  /data/env/bin/python - <<'PY'
from datasets import load_dataset
from PIL import Image
import io
ds = load_dataset('bdanko/DIV2K_train_HR', split='train')
img_col = 'hr' if 'hr' in ds.column_names else ds.column_names[0]
for i, ex in enumerate(ds):
    img = ex[img_col]
    if not isinstance(img, Image.Image):
        b = img.get('bytes') if isinstance(img, dict) else None
        img = Image.open(io.BytesIO(b)) if b else Image.open(io.BytesIO(img))
    img.convert('RGB').save('/data/datasets/DF2K/div2k_%04d.png' % i)
print('DIV2K extracted', len(ds), flush=True)
PY
  touch /data/datasets/.div2k_done
fi

# 2) Flickr2K (解压 zip 内图片)
if [ -f /data/datasets/Flickr2K.zip ] && [ ! -f /data/datasets/.flickr2k_done ]; then
  /data/env/bin/python - <<'PY'
import zipfile, io
from PIL import Image
out = '/data/datasets/DF2K'
n = 0
with zipfile.ZipFile('/data/datasets/Flickr2K.zip') as z:
    for name in z.namelist():
        if name.endswith('/'):
            continue
        low = name.lower()
        if not (low.endswith('.png') or low.endswith('.jpg') or low.endswith('.jpeg') or low.endswith('.tif')):
            continue
        try:
            img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
            img.save(out + '/flickr_%04d.png' % n)
            n += 1
        except Exception as e:
            print('skip', name, e, flush=True)
print('Flickr2K extracted', n, flush=True)
PY
  touch /data/datasets/.flickr2k_done
fi

echo DF2K_DONE
echo "DF2K image count: $(ls /data/datasets/DF2K | wc -l)"
du -sh /data/datasets/DF2K
