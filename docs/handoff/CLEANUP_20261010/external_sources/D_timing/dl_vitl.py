import os, sys, time, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
DST = Path(r'D:/models/newsrc/dinov2'); DST.mkdir(parents=True, exist_ok=True)
for name, url in [('dinov2_vitl14_pretrain.pth', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth')]:
    p = DST / name
    if p.exists() and p.stat().st_size > 1e6:
        print(f'[skip] {name}', flush=True); continue
    t0 = time.time()
    urllib.request.urlretrieve(url, p)
    print(f'[OK] {name} {p.stat().st_size/1e6:.0f}MB {time.time()-t0:.0f}s', flush=True)
print('VITL_DONE')
