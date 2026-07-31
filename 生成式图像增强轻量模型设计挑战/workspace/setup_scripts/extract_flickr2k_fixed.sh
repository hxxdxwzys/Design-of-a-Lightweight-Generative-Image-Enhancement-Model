#!/bin/bash
# 修复版 Flickr2K 提取: extractall 整体解压(顺序) + 移动图片。DIV2K 已完成跳过。
set -ex
export HF_TOKEN=$(cat /data/hf_token)

# 杀掉旧的慢提取
OLD=$(pgrep -f 'bash /data/extract_df2k.sh' | head -1)
if [ -n "$OLD" ]; then pkill -P "$OLD" 2>/dev/null; kill "$OLD" 2>/dev/null; sleep 2; echo "killed_old_extraction"; fi

mkdir -p /data/datasets/DF2K

# DIV2K (若未完成则补; 通常已完成)
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
print('DIV2K done', len(ds), flush=True)
PY
  touch /data/datasets/.div2k_done
fi

# Flickr2K: extractall + move (顺序读写, 快)
if [ -f /data/datasets/Flickr2K.zip ] && [ ! -f /data/datasets/.flickr2k_done ]; then
  /data/env/bin/python - <<'PY'
import zipfile, os, shutil
out = '/data/datasets/DF2K'
tmp = '/data/datasets/Flickr2K_raw'
os.makedirs(tmp, exist_ok=True)
print('extractall_start', flush=True)
with zipfile.ZipFile('/data/datasets/Flickr2K.zip') as z:
    z.extractall(tmp)
print('extractall_done, moving', flush=True)
n = 0
for root, _, files in os.walk(tmp):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp')):
            shutil.move(os.path.join(root, f), os.path.join(out, 'flickr_%04d_%s' % (n, f)))
            n += 1
shutil.rmtree(tmp, ignore_errors=True)
print('flickr_moved', n, flush=True)
PY
  touch /data/datasets/.flickr2k_done
fi

echo DF2K_DONE
echo "DF2K count: $(ls /data/datasets/DF2K | wc -l)"
du -sh /data/datasets/DF2K
