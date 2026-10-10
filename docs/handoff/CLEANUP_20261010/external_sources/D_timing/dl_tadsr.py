# -*- coding: utf-8 -*-
"""后台并行下载 TADSR (自包含: unet/vae/text_encoder/bert/tadsr.pkl/DAPE.pth)"""
import concurrent.futures, os, re, sys, time, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
tok = ''
for ln in Path(r'C:/生成式图像增强轻量模型设计挑战/.secrets').read_text(encoding='utf-8', errors='replace').splitlines():
    m = re.match(r'\s*HF_TOKEN\s*=\s*["\']?([^"\'\s]+)', ln)
    if m:
        tok = m.group(1); break
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('zty557/TADSR', files_metadata=True)
files = [(s.rfilename, s.size or 0) for s in info.siblings]
# 只下必要的: 权重/pkl/json/txt (跳过 doc/图)
NEED = [f for f in files if f[0].endswith(('.safetensors', '.pth', '.pkl', '.bin', '.json', '.txt'))
        and not f[0].startswith(('assets/', 'examples/'))]
print(f'[TADSR] 需要 {len(NEED)} 个文件, 合计 {sum(s for _, s in NEED)/1e9:.2f} GB', flush=True)
DST = Path(r'D:/models/newsrc/TADSR')

def get(item):
    rel, size = item
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > max(1e4, size * 0.99):
        return f'[skip] {rel}'
    url = f'https://huggingface.co/zty557/TADSR/resolve/main/{rel}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'} if tok else {})
    try:
        with urllib.request.urlopen(req, timeout=900) as r, open(out, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        return f'[OK] {rel} {out.stat().st_size/1e6:.0f}MB'
    except Exception as e:
        return f'[FAIL] {rel}: {type(e).__name__}: {str(e)[:80]}'

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(get, NEED):
        print(r, flush=True)
print('TADSR_DL_DONE')
