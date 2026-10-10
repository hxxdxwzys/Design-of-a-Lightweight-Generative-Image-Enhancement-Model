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
out.parent.mkdir(parents=True, exist_ok=True)
url = f'https://huggingface.co/zty557/TADSR/resolve/main/{rel}'
for attempt in range(4):
    try:
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'} if tok else {})
        with urllib.request.urlopen(req, timeout=1800) as r, open(out, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b)
        print(f'[OK] {rel} {out.stat().st_size/1e6:.0f}MB', flush=True); break
    except Exception as e:
        print(f'[retry {attempt+1}] {type(e).__name__}: {str(e)[:80]}', flush=True)
        time.sleep(10)
print('TADSR_UNET_DONE')
