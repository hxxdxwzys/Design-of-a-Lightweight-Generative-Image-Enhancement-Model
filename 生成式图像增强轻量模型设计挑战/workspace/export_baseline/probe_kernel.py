# -*- coding: utf-8 -*-
"""比较两种核公式顺序 + 不同尺寸下的张量化 wavelet 与参考实现的差异分布。"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))
import core as PBcore  # noqa: E402
import chain as CH  # noqa: E402
import cv2  # noqa: E402

print('--- kernel formula order ---')
k_form_a = CH.gaussian_kernel_1d(5.0).numpy().astype(np.float64)


def kernel_b(sigma=5.0, n=41):
    scale2 = -0.5 / (sigma * sigma)
    xs = np.arange(n, dtype=np.float64) - (n - 1) * 0.5
    k = np.exp(scale2 * xs * xs)
    return k / k.sum()


kb = kernel_b()
k_cv = cv2.getGaussianKernel(41, 5.0).ravel()
print('formula_a vs cv2 max=%.3e ; formula_b vs cv2 max=%.3e'
      % (np.abs(k_form_a - k_cv).max(), np.abs(kb - k_cv).max()))


def make_module(kern):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            k1 = torch.from_numpy(kern).to(torch.float32)
            self.ksize = 41
            self.register_buffer('w1', k1.view(1, 1, 1, 41).expand(3, 1, 1, 41).contiguous())
            self.register_buffer('w2', k1.view(1, 1, 41, 1).expand(3, 1, 41, 1).contiguous())

        def _blur(self, x):
            p = 20
            y = torch.nn.functional.pad(x, (p, p, 0, 0), mode='reflect')
            y = torch.nn.functional.conv2d(y, self.w1, groups=3)
            y = torch.nn.functional.pad(y, (0, 0, p, p), mode='reflect')
            return torch.nn.functional.conv2d(y, self.w2, groups=3)

        def forward(self, target, source):
            q = lambda t: torch.clamp(t * 255.0, 0.0, 255.0).to(torch.uint8)
            tn = q(torch.clamp(target, -1, 1) * 0.5 + 0.5).float() / 255.0
            sn = q(torch.clamp(source, -1, 1) * 0.5 + 0.5).float() / 255.0
            res = torch.clamp(self._blur(sn) + (tn - self._blur(tn)), 0, 1) * 255.0
            return (torch.clamp(torch.floor(res), 0, 255) / 255.0) * 2.0 - 1.0
    return M()


mods = {'A_expdiv': make_module(k_form_a), 'B_cvorder': make_module(kb)}
rng = np.random.default_rng(7)
print('--- e2e per shape (max_abs, frac_pixels_diff) ---')
for shape in [(1, 3, 64, 64), (1, 3, 97, 79), (1, 3, 128, 128), (1, 3, 200, 151), (1, 3, 512, 512)]:
    tg = torch.from_numpy(rng.random(shape).astype(np.float32)) * 2 - 1
    sr = torch.from_numpy(rng.random(shape).astype(np.float32)) * 2 - 1
    ref = PBcore.wavelet_color_fix_tensor(tg, sr, sigma=5)
    for name, m in mods.items():
        got = m(tg, sr)
        d = (ref - got).abs()
        nd = int((d > 0).sum())
        print('  %-16s %-10s max=%.3e frac_diff=%.3e' % (shape, name, float(d.max()), nd / d.numel()))
