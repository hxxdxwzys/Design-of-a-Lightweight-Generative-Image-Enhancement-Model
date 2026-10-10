import os, re, sys, time, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
tok = ''
for ln in Path(r'C:/生成式图像增强轻量模型设计挑战/.secrets').read_text(encoding='utf-8', errors='replace').splitlines():
    m = re.match(r'\s*HF_TOKEN\s*=\s*["\']?([^"\'\s]+)', ln)
    if m: tok = m.group(1); break
rel = 'unet/diffusion_pytorch_model.safetensors'
out = Path(r'D:/models/newsrc/TADSR') / rel
TARGET = 3463_700_000   # 约 3.46GB
url = f'https://huggingface.co/zty557/TADSR/resolve/main/{rel}'
for attempt in range(8):
    have = out.stat().st_size if out.exists() else 0
    if have >= TARGET * 0.999:
        print(f'[done] {have/1e6:.0f}MB', flush=True); break
    hdr = {'Authorization': f'Bearer {tok}'} if tok else {}
    if have:
        hdr['Range'] = f'bytes={have}-'
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=600) as r, open(out, 'ab' if have else 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b)
        print(f'[chunk] {out.stat().st_size/1e6:.0f}MB', flush=True)
    except Exception as e:
        print(f'[retry {attempt+1}] have={have/1e6:.0f}MB {type(e).__name__}: {str(e)[:70]}', flush=True)
        time.sleep(5)
print(f'[final] {out.stat().st_size/1e6:.0f}MB', flush=True)
print('TADSR_UNET2_DONE')
