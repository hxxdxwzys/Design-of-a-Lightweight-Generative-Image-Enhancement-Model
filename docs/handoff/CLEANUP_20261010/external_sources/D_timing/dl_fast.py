# -*- coding: utf-8 -*-
"""用 HF token + 并行直链下载 (比 hf hub 快 10x)"""
import concurrent.futures, os, re, sys, time, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)

tok = ''
for ln in Path(r'C:/生成式图像增强轻量模型设计挑战/.secrets').read_text(encoding='utf-8', errors='replace').splitlines():
    m = re.match(r'\s*HF_TOKEN\s*=\s*["\']?([^"\'\s]+)', ln)
    if m:
        tok = m.group(1); break
print('token:', 'yes' if tok else 'no', flush=True)

JOBS = [
    ('VCLab-PolyU/PixRestore', 'pixrestore-xl/clean_weights/ema_model.safetensors', r'D:/models/newsrc/PixRestore'),
    ('VCLab-PolyU/PixRestore', 'pixrestore-l/clean_weights/ema_model.safetensors', r'D:/models/newsrc/PixRestore'),
    ('VCLab-PolyU/PixRestore', 'pixrestore-b/clean_weights/ema_model.safetensors', r'D:/models/newsrc/PixRestore'),
    ('VCLab-PolyU/PixRestore', 'pixrestore-s/clean_weights/ema_model.safetensors', r'D:/models/newsrc/PixRestore'),
]


def get(args):
    repo, rel, dst = args
    out = Path(dst) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1e5:
        return f'[skip] {rel} {out.stat().st_size/1e6:.0f}MB'
    url = f'https://huggingface.co/{repo}/resolve/main/{rel}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'} if tok else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(out, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        return f'[OK] {rel} {out.stat().st_size/1e6:.0f}MB {time.time()-t0:.0f}s'
    except Exception as e:
        return f'[FAIL] {rel}: {type(e).__name__}: {str(e)[:100]}'


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(get, JOBS):
        print(r, flush=True)
print('DL2_DONE')
