# -*- coding: utf-8 -*-
"""q7_postproc_probe.py — 测 Q 代理(原生 4K 的 maniqa/clipiqa/niqe)对像素后处理的响应.
目的: 判断"提高 Q"是否存在像素级杠杆 (旧世界用旧模型判过"像素后处理"死, 但那是 lpips 口径).
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch, pyiqa
from PIL import Image, ImageFilter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
dev = 'cuda'
iq = {k: pyiqa.create_metric(k, device=dev) for k in ('maniqa', 'clipiqa', 'niqe')}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:15]]

def variants(im):
    out = {'A_M24': im}
    out['B_med3'] = im.filter(ImageFilter.MedianFilter(3))
    m3 = out['B_med3']
    out['C_med3_us'] = m3.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    g = im.filter(ImageFilter.GaussianBlur(0.8))
    out['D_g08_us'] = g.filter(ImageFilter.UnsharpMask(radius=3, percent=180, threshold=2))
    out['E_med5_us'] = im.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    return out

acc = {}
t0 = time.time()
for nm in names:
    vs = variants(Image.open(M24D / (nm + '.jpg')).convert('RGB'))
    for k, im in vs.items():
        t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
        d = acc.setdefault(k, {kk: [] for kk in iq})
        for kk, m in iq.items():
            with torch.no_grad():
                d[kk].append(float(m(t).item()))
        del t
    print('  ..%s (%.0fs)' % (nm, time.time() - t0), flush=True)
print('=== 结果 (原生 4K, %d 张) ===' % len(names))
base = None
for k, d in acc.items():
    r = {kk: float(np.mean(v)) for kk, v in d.items()}
    if base is None: base = r
    print('%-12s maniqa=%.4f (%+.4f)  clipiqa=%.4f (%+.4f)  niqe=%.4f (%+.4f)' % (
        k, r['maniqa'], r['maniqa'] - base['maniqa'], r['clipiqa'], r['clipiqa'] - base['clipiqa'],
        r['niqe'], r['niqe'] - base['niqe']))
Path(r'D:/tmp_timing/q7_postproc.json').write_text(json.dumps(acc, indent=1), encoding='utf-8')
print('Q7_DONE')
