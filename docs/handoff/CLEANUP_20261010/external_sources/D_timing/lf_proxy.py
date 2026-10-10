# -*- coding: utf-8 -*-
"""低频保真度代理 (无 GT): LFdiff = mean|down(output) - down(LQ)|
原理: LQ = 退化(GT), 而模糊/噪声/JPEG 基本不动低频 -> LQ 的低频 ~= GT 的低频
      => 我们的输出与 LQ 的低频差异, 近似"我们与 test GT 的低频差异" (= 有参指标的盲区代理)
"""
import numpy as np
from pathlib import Path
from PIL import Image

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = [
    ('M24           43.6951', r'D:/models/final_compliant/output_dir', 43.6951),
    ('M24_v4        43.5844', r'D:/models/final_compliant/output_dir', 43.5844),
    ('probeA        43.2837', r'D:/models/final_compliant_probeA_sc04/output_dir', 43.2837),
    ('ProbeDetail   43.0699', r'D:/models/pkg_ProbeDetail/output_dir', 43.0699),
    ('M24sc00       37.6316', r'D:/models/final_compliant_M24sc00/output_dir', 37.6316),
    ('M24p4          7.1829', r'D:/models/final_compliant_M24p4/output_dir', 7.1829),
    ('TinyV1        27.3521', r'D:/models/pkg_TinyV1/output_dir', 27.3521),
    ('TinyV2        29.4225', r'D:/models/pkg_TinyV2/output_dir', 29.4225),
    ('TinyV4          (未交)', r'D:/models/pkg_TinyV4/output_dir', None),
    ('HYPIR           (未交)', r'D:/models/hypir/out_test100/result', None),
]
IDX = list(range(1, 101, 4))   # 25 张
SCALE = 8                       # 降到 1/8 看低频

def lf(arr):
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w // SCALE, h // SCALE), Image.BOX), dtype=np.float32)

lqs = {}
for i in IDX:
    p = TEST / f'case{i}.jpg'
    if p.exists():
        lqs[i] = lf(np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8))

rows = []
for name, d, score in PKGS:
    D = Path(d); diffs = []
    for i in IDX:
        if i not in lqs: continue
        for ext in ('.jpg', '.png'):
            p = D / f'case{i}{ext}'
            if p.exists():
                a = lf(np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8))
                if a.shape == lqs[i].shape:
                    diffs.append(float(np.abs(a - lqs[i]).mean()))
                break
    if diffs:
        rows.append((name, float(np.mean(diffs)), score))
print(f'{"包":24s} {"LFdiff(越低越接近GT低频)":>22s} {"官方分":>10s}')
for n, v, s in sorted(rows, key=lambda r: r[1]):
    print(f'{n:24s} {v:22.3f} {("%.4f" % s) if s else "-":>10s}')
print()
sc = [(r[1], r[2]) for r in rows if r[2]]
if len(sc) > 2:
    import math
    a = np.array(sc)
    print('相关性 (LFdiff vs 官方分): Pearson r = %.3f' % float(np.corrcoef(a[:,0], a[:,1])[0,1]))
print('LFPROXY_DONE')
