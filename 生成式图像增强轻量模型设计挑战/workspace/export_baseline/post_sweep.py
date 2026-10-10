# -*- coding: utf-8 -*-
"""后处理参数扫描（CPU 应用）：在 f_v7sc0.8（OPT-01 的 4 张全 4K）上生成多个 unsharp 变体。"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
JPEG = {'quality': 98, 'subsampling': 0, 'optimize': False, 'progressive': False}


def gk(sigma, device, dtype):
    ks = int(round(sigma * 3)) * 2 + 1
    x = torch.arange(ks, dtype=torch.float64) - (ks - 1) / 2
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return (k / k.sum()).to(device=device, dtype=dtype)


def unsharp(t, sigma=1.2, percent=60.0, thr=3.0):
    k = gk(sigma, t.device, t.dtype)
    p = k.numel() // 2
    c = int(t.shape[1])
    y = F.pad(t, (p, p, 0, 0), mode='reflect')
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, p, p), mode='reflect')
    blur = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    d = t - blur
    m = (d.abs() > thr / 255.0 * 2.0).to(t.dtype)
    out = (t + percent / 100.0 * d * m).clamp(-1, 1)
    u8 = torch.round((out + 1.0) * 127.5)
    return (u8 / 127.5) - 1.0


VARIANTS = {
    'u_s1.0_p60_t3': dict(sigma=1.0, percent=60.0, thr=3.0),
    'u_s1.2_p80_t3': dict(sigma=1.2, percent=80.0, thr=3.0),
    'u_s1.2_p100_t3': dict(sigma=1.2, percent=100.0, thr=3.0),
    'u_s1.5_p80_t4': dict(sigma=1.5, percent=80.0, thr=4.0),
    'u_s1.2_p60_t2': dict(sigma=1.2, percent=60.0, thr=2.0),
    'u_s1.2_p140_t3': dict(sigma=1.2, percent=140.0, thr=3.0),
}
if __name__ == '__main__':
    which = sys.argv[1:] or list(VARIANTS)
    src = Path(r'D:\models\iqa_screen\f_v7sc0.8')
    for name in which:
        par = VARIANTS[name]
        out = Path(r'D:\models\iqa_screen') / name
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob('*.jpg')):
            arr = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            y = unsharp(t, **par)
            u8 = ((y + 1.0) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).numpy()
            Image.fromarray(u8).save(out / p.name, format='JPEG', **JPEG)
        print('made', name, par, flush=True)
