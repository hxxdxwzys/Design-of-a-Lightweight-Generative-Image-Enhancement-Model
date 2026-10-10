# -*- coding: utf-8 -*-
"""下载 PixRestore 需要的 DINOv2 ViT-S/14 + 代码 (torch hub 可能被墙, 直接下权重)"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.pop('HF_HUB_OFFLINE', None)
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
from pathlib import Path
import urllib.request

DST = Path(r'D:/models/newsrc/dinov2'); DST.mkdir(parents=True, exist_ok=True)
URLS = {
    'dinov2_vits14_pretrain.pth': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'dinov2_vitb14_pretrain.pth': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
}
for name, url in URLS.items():
    p = DST / name
    if p.exists() and p.stat().st_size > 1e6:
        print(f'[skip] {name} {p.stat().st_size/1e6:.1f}MB', flush=True); continue
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, p)
        print(f'[OK] {name} {p.stat().st_size/1e6:.1f}MB {(time.time()-t0):.0f}s', flush=True)
    except Exception as e:
        print(f'[FAIL] {name}: {type(e).__name__}: {str(e)[:120]}', flush=True)
print('DINO_DONE')
