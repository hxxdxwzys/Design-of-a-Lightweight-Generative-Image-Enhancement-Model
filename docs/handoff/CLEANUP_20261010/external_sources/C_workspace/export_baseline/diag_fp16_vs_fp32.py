# -*- coding: utf-8 -*-
"""诊断：fp16 eager 4K 输出 vs 已验收 fp32（R-B）输出的差异结构。仅 CPU 读盘，不用 GPU。"""
import numpy as np
from pathlib import Path

RUN = Path(r'C:\models\exports\EXPORT01_RUN_20260917_135055')
SRC = RUN / 'evidence' / 'src_outputs'
RB = Path('D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01')

pairs = [
    ('case1_full', SRC / 'case1_full_src.npy', RB / 'attempt_01_072821_A_case1_full_r2.npy'),
    ('case7_full', SRC / 'case7_full_src.npy', RB / 'attempt_01_072821_A_case7_full_r3.npy'),
    ('case1_crop512', SRC / 'case1_crop512_src.npy',
     RB / 'attempt_01_072821_A_case1_crop512_r1.npy'),
]
for name, a, b in pairs:
    x = np.load(a, allow_pickle=False)
    y = np.load(b, allow_pickle=False)
    print('==', name, x.shape, y.shape, x.dtype, y.dtype)
    if x.shape != y.shape:
        print('   shape mismatch -> skip'); continue
    d = np.abs(x - y)
    print('   max=%.4f mean=%.5f frac>0.05=%.4f' % (d.max(), d.mean(), (d > 0.05).mean()))
    dm = d.mean(axis=(0, 1))          # (H, W)
    cm = dm.mean(axis=0)              # per column
    rm = dm.mean(axis=1)              # per row
    print('   col-mean: first8=%s  last8=%s  max=%.4f' %
          (np.round(cm[:8], 4).tolist(), np.round(cm[-8:], 4).tolist(), cm.max()))
    print('   row-mean: first8=%s  last8=%s  max=%.4f' %
          (np.round(rm[:8], 4).tolist(), np.round(rm[-8:], 4).tolist(), rm.max()))
    # 低频/周期结构：列均值的 FFT 峰值
    F = np.abs(np.fft.rfft(cm - cm.mean()))
    k = int(np.argmax(F[1:]) + 1)
    print('   col FFT peak k=%d (period=%.1f px) amp=%.4f' % (k, len(cm) / k, F[k]))
    # 平移搜索（降采样 8 倍）
    xs = x[:, :, ::8, ::8]
    best = None
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            ys = y[:, :, ::8, ::8]
            ys = np.roll(np.roll(ys, dy, axis=2), dx, axis=3)
            m = float(np.abs(xs - ys).mean())
            if best is None or m < best[0]:
                best = (m, dy, dx)
    print('   best shift (mean,dy,dx)=%s ; no-shift=%.5f' % (best, d.mean()))
