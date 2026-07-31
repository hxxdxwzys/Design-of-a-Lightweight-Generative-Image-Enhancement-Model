#!/bin/bash
# v4 数据下载 (上海, hf-mirror): SD2.1 (vae+unet+text_encoder+tokenizer+scheduler) + DF2K
set -ex
export HF_TOKEN=$(cat /data/hf_token)
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data/weights /data/datasets/DF2K

# 1) SD2.1-base (safetensors)
if [ ! -f /data/weights/sd21base/.done ]; then
  /data/env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Manojb/stable-diffusion-2-1-base', local_dir='/data/weights/sd21base',
    allow_patterns=['vae/*','unet/*.safetensors','unet/config.json',
                    'text_encoder/*.safetensors','text_encoder/config.json',
                    'tokenizer/*','scheduler/*'],
    ignore_patterns=['*.bin','*.fp16.*'], max_workers=8)
print('SD21 done', flush=True)
PY
  touch /data/weights/sd21base/.done
fi

# 2) DIV2K (parquet -> PNG)
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

# 3) Flickr2K (zip -> extractall + move)
if [ ! -f /data/datasets/.flickr2k_done ]; then
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o /data/datasets/Flickr2K.zip \
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

echo V4_DATA_DONE
echo "DF2K count: $(ls /data/datasets/DF2K | wc -l)"
du -sh /data/weights/sd21base /data/datasets/DF2K
