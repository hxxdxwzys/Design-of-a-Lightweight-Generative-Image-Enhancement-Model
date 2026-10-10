# -*- coding: utf-8 -*-
"""image_term.py (2026-09-15) — 把官方分拆成【速度项】(已由同图对实测标定) + 【图像项】

速度曲线(相对 S(0.78x)=0), 由三组"图像逐像素/逐字节相同、只改时延"的读数标定:
  M24(0.78x)->avg2(0.382x) = -7.05 / 1.031 翻倍 = 6.84 分/翻倍  (1x 以下)
  M24(0.78x)->Slow8(0.098x) = -14.94 / 2.995 翻倍 = 4.99 分/翻倍 (1x 以下)
  TinyV2(19.5x)->TinyV2g(53.9x, 图逐像素相同) = +3.86 / 1.467 翻倍 = 2.63 分/翻倍 (1x 以上)
=> 本脚本用 [1x 以下 5.0, 1x 以上 2.63] 两段斜率, 并同时给出 [4.99, 6.84] 的区间看稳健性。
图像项 IT = 官方分 - S(ratio);  若"图像项只是观感的函数", 则同一批图在不同速度下 IT 应相同(自检)。
"""
import json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# tag, 官方分, 加速比, 备注, 我们已知的 test30 clipiqa(原生 4K 口径)
ROWS = [
    ('M24',           43.6951, 0.78,  '0.5B VOSR + LoRA sc0.8', 0.9023),
    ('probeA_sc04',   43.2837, 0.78,  'LoRA sc0.4',             0.7700),
    ('ProbeDetail',   43.0699, 0.78,  'M24 图 + 细节 x2.27',     None),
    ('ProbeBlend',    40.5354, 0.78,  'M24 图向 LQ 回退',        None),
    ('M24sc00',       37.6316, 0.78,  '去 LoRA 发糊',           0.6448),
    ('M24avg2',       36.6414, 0.382, '同 M24 图, 2 样本平均',   0.9023),
    ('M24p4',          7.1829, 0.975, '剪枝毁图',                None),
    ('ProbeSlow8',    28.7538, 0.098, '同 M24 图, 8x 慢',        0.9023),
    ('TinyV7gan',     41.7346, 61.9,  '256域+GAN 0.87M',        0.5059),
    ('TinyV5d',       33.0382, 61.4,  '同架构无 GAN',           0.3897),
    ('TinyV2g',       33.2872, 53.9,  'GT 目标 + graph',         0.3532),
    ('TinyV2',        29.4225, 19.5,  '同 V2g 图, 慢速档',        0.3532),
    ('TinyV1',        27.3521, 6.1,   '512 域无 GAN',           0.3367),
    ('PixB',          18.9710, 1.5,   'PixRestore-b',           0.3143),
    ('PixB@v9brun',   None,    12.4,  '待定件: M24 图 + v9b runner', 0.9023),
    ('TinyV9b',       None,    12.4,  '512 域 0.38M 待评测',      0.6409),
]
SLOPE_LO, SLOPE_HI = 5.0, 2.63


def S(r, slope_lo=SLOPE_LO):
    if r is None:
        return None
    x = np.log2(r / 0.78)
    return (SLOPE_HI if r >= 0.78 else slope_lo) * x


print('%-14s %8s %7s %8s | %8s %8s | %8s' % ('tag', 'score', 'ratio', 'S(ratio)', '图项IT', 'IT(陡)', 'clipiqa'))
for tag, sc, r, note, cq in ROWS:
    s = S(r)
    if sc is None:
        print('%-14s %8s %7s %8.2f | %8s %8s | %8s   %s' % (tag, '-', r, s, '-', '-', cq, note))
        continue
    print('%-14s %8.4f %7s %8.2f | %8.2f %8.2f | %8s   %s' % (
        tag, sc, r, s, sc - s, sc - S(r, 6.84), cq, note))

scored = [(t, s, r, cq) for t, s, r, _, cq in ROWS if s is not None and cq is not None]
print('\n=== 图像项 IT vs 已知 clipiqa(原生 4K) ===')
for t, s, r, cq in sorted(scored, key=lambda z: z[3]):
    print('  clipiqa=%.4f  IT=%6.2f   %s(r=%s)' % (cq, s - S(r), t, r))
xs = np.array([z[3] for z in scored]); ys = np.array([z[1] - S(z[2]) for z in scored])
print('  Pearson r = %+.3f   (n=%d)' % (float(np.corrcoef(xs, ys)[0, 1]), len(scored)))

# 若面板已出: 每个(口径,指标) 与 IT 的相关(用未经拟合的 S 曲线, 避免共线)
pj = Path('D:/tmp_timing/diag_panel.json')
if pj.exists():
    panel = json.loads(pj.read_text(encoding='utf-8'))
    print('\n=== 面板: IT(由已标定 S 曲线反推) vs 各口径指标 ===')
    for mode in ('native', 'ms1024', 'ms512', 'c1024'):
        for m in ('clipiqa', 'musiq', 'maniqa', 'niqe', 'lapvar'):
            pts = []
            for t, s, r, note, cq in ROWS:
                if s is None or t not in panel or mode not in panel[t]:
                    continue
                v = panel[t][mode].get(m)
                if v is None:
                    continue
                pts.append((v, s - S(r)))
            if len(pts) >= 6:
                x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
                print('  %-7s %-8s n=%2d  Pearson r=%+.3f' % (mode, m, len(pts), float(np.corrcoef(x, y)[0, 1])))
else:
    print('\n(面板 diag_panel.json 尚未生成)')
print('IMAGE_TERM_DONE')
