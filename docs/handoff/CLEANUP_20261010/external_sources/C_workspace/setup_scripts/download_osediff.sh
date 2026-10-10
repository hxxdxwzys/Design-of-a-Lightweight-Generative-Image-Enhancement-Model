#!/bin/bash
# 下载 OSEDiff 依赖：代码仓(含osediff.pkl/DAPE.pth) + SD2.1-base + RAM
set -ex
mkdir -p /data/weights /data/osediff

# 1) OSEDiff 仓库 (含 osediff.pkl, DAPE.pth, 代码)
if [ ! -d /data/osediff/OSEDiff/.git ]; then
  git clone https://github.com/cswry/OSEDiff.git /data/osediff/OSEDiff
fi

# 2) SD2.1-base (VAE/UNet/text-encoder/tokenizer)，多源回退
if [ ! -f /data/weights/sd21base/.done ]; then
  /data/env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
ok = False
for repo in ['Manojb/stable-diffusion-2-1-base', 'stabilityai/stable-diffusion-2-1', 'stabilityai/stable-diffusion-2-base']:
    try:
        print('trying', repo, flush=True)
        snapshot_download(repo, local_dir='/data/weights/sd21base', max_workers=8)
        print('OK from', repo, flush=True); ok = True; break
    except Exception as e:
        print('FAIL', repo, repr(e)[:200], flush=True)
assert ok, 'all SD21 repos failed'
PY
  touch /data/weights/sd21base/.done
fi

# 3) RAM (ram_swin_large_14m)
if [ ! -f /data/weights/ram_swin_large_14m.pth ]; then
  curl -L --retry 3 -o /data/weights/ram_swin_large_14m.pth \
    https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
fi

echo DOWNLOAD_DONE
echo "=== sizes ==="
du -sh /data/weights/* 2>/dev/null
ls -lh /data/osediff/OSEDiff/preset/models/ 2>/dev/null
