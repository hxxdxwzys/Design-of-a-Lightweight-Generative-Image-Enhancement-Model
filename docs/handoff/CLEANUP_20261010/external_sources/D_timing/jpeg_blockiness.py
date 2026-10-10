# -*- coding: utf-8 -*-
"""JPEG 块效应检测 v2 (2026-08-14, CPU only)
指标: 8x8 块边界处水平/垂直 |差分| 均值 与 块内 |差分| 均值的比值 (clean≈1.0, 块效应越强越大)
标定: 3 张 DIV2K HQ, jpeg(q) q∈[30..90] -> 同一指标 -> 反解每张测试图原始 q 估计
输出: D:/tmp_timing/jpeg_blockiness.json (覆盖 v1)
"""
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
HQ_DIR = Path(r'D:/datasets/div2k/pngtxt_ori')
OUT = Path(r'D:/tmp_timing/jpeg_blockiness.json')


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def blockiness_ratio(img):
    """边界差分 / 块内差分 的比值 - 1; 越大块效应越强"""
    g = np.asarray(img.convert('L'), dtype=np.float32)
    h, w = g.shape
    if h < 16 or w < 16:
        return 0.0
    dh = np.abs(g[:, 1:] - g[:, :-1])
    bh = (np.arange(w - 1) % 8 == 7)
    mh_b = dh[:, bh].mean() if bh.any() else 0.0
    mh_w = dh[:, ~bh].mean() if (~bh).any() else 0.0
    dv = np.abs(g[1:, :] - g[:-1, :])
    bv = (np.arange(h - 1) % 8 == 7)
    mv_b = dv[bv, :].mean() if bv.any() else 0.0
    mv_w = dv[~bv, :].mean() if (~bv).any() else 0.0
    ratio = (mh_b / max(mh_w, 1e-9) + mv_b / max(mv_w, 1e-9)) / 2.0
    return float(ratio - 1.0)


def main():
    t0 = time.time()
    hq = sorted(HQ_DIR.glob('*.png'))[:3]
    qs = [30, 40, 50, 60, 70, 80, 90, 95]
    cal = {q: [] for q in qs}
    for p in hq:
        base = Image.open(p).convert('RGB')
        for q in qs:
            cal[q].append(blockiness_ratio(_jpeg(base, q)))
    qmap = {q: float(np.mean(v)) for q, v in cal.items()}
    print('[calib] jpeg q -> blockiness_ratio:', flush=True)
    for q in qs:
        print(f'    q={q}: {qmap[q]:.4f}', flush=True)

    def inv(x):
        ks = sorted(qs)
        vs = [qmap[k] for k in ks]  # 值随 q 递减
        if x >= vs[0]:
            return ks[0]
        if x <= vs[-1]:
            return ks[-1]
        for i in range(len(vs) - 1):
            if vs[i] >= x >= vs[i + 1]:
                frac = (vs[i] - x) / max(vs[i] - vs[i + 1], 1e-12)
                return ks[i] + frac * (ks[i + 1] - ks[i])
        return ks[-1]

    rows = []
    for tag, d in [('test', TEST_DIR), ('val', VAL_DIR)]:
        files = sorted(d.glob('*.jpg'))
        if tag == 'val':
            files = [f for f in files if '_lq' in f.name]
        for p in files:
            b = blockiness_ratio(Image.open(p).convert('RGB'))
            rows.append({'tag': tag, 'name': p.name, 'blockiness_ratio': round(b, 5), 'est_jpeg_q': round(inv(b), 1)})

    test_r = [r for r in rows if r['tag'] == 'test']
    val_r = [r for r in rows if r['tag'] == 'val']
    bt = np.array([r['blockiness_ratio'] for r in test_r])
    qt = np.array([r['est_jpeg_q'] for r in test_r])
    bv_ = np.array([r['blockiness_ratio'] for r in val_r])

    def st(a):
        return {'n': len(a), 'mean': round(float(a.mean()), 4), 'median': round(float(np.median(a)), 4),
                'p10': round(float(np.percentile(a, 10)), 4), 'p90': round(float(np.percentile(a, 90)), 4),
                'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4)}

    summary = {'test_blockiness_ratio': st(bt), 'test_est_jpeg_q': st(qt),
               'val_blockiness_ratio': st(bv_), 'calibration': {str(k): round(v, 4) for k, v in qmap.items()}}
    print('\n===== 汇总 =====', flush=True)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print('test est_jpeg_q histogram (<=40 / 40-60 / 60-80 / >80):',
          np.histogram(qt, bins=[0, 40, 60, 80, 101])[0].tolist(), flush=True)
    with open(OUT, 'w') as f:
        json.dump({'per_image': rows, 'summary': summary}, f, indent=1)
    print(f'[saved] {OUT} ({(time.time() - t0) / 60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
