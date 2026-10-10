# -*- coding: utf-8 -*-
import os, sys, time
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = 'D:/models/final_compliant/output_dir/case1.jpg'
im = Image.open(p).convert('RGB')
W, H = im.size
print('img', im.size, flush=True)
variants = [('4K', im),
            ('crop1024', im.crop(((W-1024)//2, (H-1024)//2, (W+1024)//2, (H+1024)//2))),
            ('resize512', im.resize((512, 512), Image.BICUBIC))]
for name in ['clipiqa', 'musiq', 'maniqa', 'niqe']:
    try:
        t0 = time.time(); m = pyiqa.create_metric(name, device='cuda')
        print('[create] %-8s %.1fs' % (name, time.time()-t0), flush=True)
    except Exception as e:
        print('[create] %-8s EXC %s: %s' % (name, type(e).__name__, str(e)[:200]), flush=True)
        continue
    for tag, arr in variants:
        try:
            t = torch.from_numpy(np.asarray(arr)).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
            t0 = time.time()
            with torch.no_grad(): v = float(m(t).item())
            print('  %-8s %-10s %.4f  %.1fs' % (name, tag, v, time.time()-t0), flush=True)
            del t
        except Exception as e:
            print('  %-8s %-10s EXC %s: %s' % (name, tag, type(e).__name__, str(e)[:200]), flush=True)
    del m
    torch.cuda.empty_cache()
print('TIMING_IQA_DONE', flush=True)
