# -*- coding: utf-8 -*-
"""Q1b: 用可精确表示 x.5 的值区分 round-half-up 与 trunc；Q4b: 用 trunc 重算端到端。"""
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

cands = [0.5, 1.5, 2.5, 127.5, 200.5, 254.5]
ts = []
for c in cands:
    t = np.float32(c / 255.0)
    prod = float(np.float64(t) * 255.0)
    ts.append((c, t, prod))
print('--- candidate exact .5 probes (c, float32 t, exact t*255) ---')
for c, t, prod in ts:
    print('  c=%-6s t=%.12g  t*255=%.12g  exact_half=%s' % (c, float(t), prod, abs(prod - c) < 1e-12))
t = torch.tensor([[[x[1] for x in ts]]], dtype=torch.float32).repeat(3, 1, 1).contiguous()
pil = np.array(to_pil_image(t))[0].tolist()
print('PIL out            =', pil)
print('round_half_up pred =', torch.clamp(t * 255.0 + 0.5, 0, 255).to(torch.uint8)[0, 0].tolist())
print('trunc pred         =', torch.clamp(t * 255.0, 0, 255).to(torch.uint8)[0, 0].tolist())

# ── Q4b: 端到端，两种量化规则各算一遍 ──
PB = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\pipeline_baseline')
sys.path.insert(0, str(PB))
import core as PBcore  # noqa: E402
import cv2  # noqa: E402

sigma = 5.0
ks = int(round(sigma * 4 * 2 + 1)) | 1
xs = torch.arange(ks, dtype=torch.float64) - (ks - 1) / 2.0
k1 = (torch.exp(-(xs ** 2) / (2 * sigma ** 2)))
k1 = (k1 / k1.sum()).to(torch.float32)


def blur(x):  # x: torch [1,3,H,W] float32
    w1 = k1.view(1, 1, 1, ks).expand(3, 1, 1, ks).contiguous()
    w2 = k1.view(1, 1, ks, 1).expand(3, 1, ks, 1).contiguous()
    y = torch.nn.functional.pad(x, (ks // 2, ks // 2, 0, 0), mode='reflect')
    y = torch.nn.functional.conv2d(y, w1, groups=3)
    y = torch.nn.functional.pad(y, (0, 0, ks // 2, ks // 2), mode='reflect')
    return torch.nn.functional.conv2d(y, w2, groups=3)


def wavelet_torch(target, source, quant):
    q = (lambda t: torch.clamp(t * 255.0 + 0.5, 0, 255).to(torch.uint8)) if quant == 'round' \
        else (lambda t: torch.clamp(t * 255.0, 0, 255).to(torch.uint8))
    tn = q(torch.clamp(target, -1, 1) * 0.5 + 0.5).float() / 255.0
    sn = q(torch.clamp(source, -1, 1) * 0.5 + 0.5).float() / 255.0
    res = torch.clamp(blur(sn) + (tn - blur(tn)), 0, 1) * 255.0
    return (torch.clamp(torch.floor(res), 0, 255) / 255.0) * 2.0 - 1.0


rng = np.random.default_rng(1)
cases = {
    'random': (torch.from_numpy(rng.random((1, 3, 96, 80)).astype(np.float32)) * 2 - 1,
               torch.from_numpy(rng.random((1, 3, 96, 80)).astype(np.float32)) * 2 - 1),
    'gray': (torch.full((1, 3, 64, 64), -0.2), torch.full((1, 3, 64, 64), 0.3)),
    'grad': (torch.linspace(-1, 1, 96 * 80).reshape(1, 1, 96, 80).repeat(1, 3, 1, 1),
             torch.linspace(1, -1, 96 * 80).reshape(1, 1, 96, 80).repeat(1, 3, 1, 1)),
}
for name, (tg, sr) in cases.items():
    ref = PBcore.wavelet_color_fix_tensor(tg, sr, sigma=5)
    for quant in ('round', 'trunc'):
        got = wavelet_torch(tg, sr, quant)
        d = (ref - got).abs()
        u8_ref = torch.clamp((ref + 1) / 2 * 255 + 0.5, 0, 255).to(torch.uint8)
        u8_got = torch.clamp((got + 1) / 2 * 255 + 0.5, 0, 255).to(torch.uint8)
        print('Q4b %-7s %-6s max_abs=%.3e mean=%.3e uint8_identical=%s'
              % (name, quant, float(d.max()), float(d.mean()), bool(torch.equal(u8_ref, u8_got))))
