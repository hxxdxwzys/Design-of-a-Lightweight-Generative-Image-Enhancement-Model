# -*- coding: utf-8 -*-
"""下载 PixRestore 权重 (4 个尺寸, 共 ~4.8GB) + VARestorer 权重"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.pop('HF_HUB_OFFLINE', None)
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
from huggingface_hub import snapshot_download

JOBS = [
    ('VCLab-PolyU/PixRestore', r'D:/models/newsrc/PixRestore'),
    ('EvanEternal/VARestorer', r'D:/models/newsrc/VARestorer'),
]
for rid, dst in JOBS:
    t0 = time.time()
    print(f'>>> 下载 {rid} -> {dst}', flush=True)
    try:
        p = snapshot_download(repo_id=rid, local_dir=dst,
                              allow_patterns=['*.safetensors', '*.json', '*.md', '*.pth', '*.py'])
        tot = sum(f.stat().st_size for f in __import__('pathlib').Path(dst).rglob('*') if f.is_file())
        print(f'[OK] {rid}  {tot/1e9:.2f} GB  {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception as e:
        print(f'[FAIL] {rid}: {type(e).__name__}: {str(e)[:200]}', flush=True)
print('DL_DONE')
