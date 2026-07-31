#!/bin/bash
# 下载 OSEDiff 依赖 v2 (tarball 代替 git)：代码仓(含osediff.pkl/DAPE.pth) + SD2.1-base + RAM
set -ex
mkdir -p /data/weights /data/osediff

# 1) OSEDiff 仓库 via tarball (无需 git)
if [ ! -f /data/osediff/OSEDiff/osediff.py ]; then
  curl -L --retry 3 -o /tmp/osediff.tar.gz https://github.com/cswry/OSEDiff/archive/refs/heads/main.tar.gz
  tar xzf /tmp/osediff.tar.gz -C /data/osediff
  mv /data/osediff/OSEDiff-main /data/osediff/OSEDiff
  rm -f /tmp/osediff.tar.gz
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
