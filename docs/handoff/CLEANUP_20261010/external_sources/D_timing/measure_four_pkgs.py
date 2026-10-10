# -*- coding: utf-8 -*-
"""四个已出分包的多指标测量 (2026-09-13) —— 用分数跨度 7~44 的新数据重做回归
动机:
  M24p4   = 7.1829   (DiT 剪到 4/28, 图像被毁)
  M24sc00 = 37.6316  (纯基座, 中等质量损失)
  probeA  = 43.2837  (小幅质量变化)
  M24     = 43.6951  (基准)
  分数跨度 36.5 分 -> 这次回归终于有判别力, 能找出官方到底认哪个指标。
指标: niqe / clipiqa / musiq / maniqa  (测试集 sorted[:30] 同名配对, 增量写 JSONL)
用法: python measure_four_pkgs.py [--n 30]
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DIRS = [
    ('M24',     r'D:/models/final_compliant/output_dir',            43.6951),
    ('probeA',  r'D:/models/final_compliant_probeA_sc04/output_dir', 43.2837),
    ('sc00',    r'D:/models/final_compliant_M24sc00/output_dir',     37.6316),
    ('p4',      r'D:/models/final_compliant_M24p4/output_dir',        7.1829),
    ('avg2',    r'D:/models/final_compliant_M24avg2/output_dir',     None),
]
KEYS = ('niqe', 'clipiqa', 'musiq')   # 砍掉 MANIQA (4K 上 ~10min/张, 全量要 50h; 且旧包没有 maniqa 数据可比)
JSONL = Path(r'D:/tmp_timing/four_pkgs_metrics.jsonl')
OUT = Path(r'D:/tmp_timing/four_pkgs_metrics.json')


class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb():
    st = MS(); st.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ms = {k: pyiqa.create_metric(k, device=dev) for k in KEYS}
    names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
    print(f'[start] {len(names)} 张 x {len(DIRS)} 包 x {len(KEYS)} 指标 | {dev} | '
          f'可用内存 {free_gb():.1f} GB', flush=True)

    done = set()
    if JSONL.exists():
        for ln in JSONL.read_text(encoding='utf-8').splitlines():
            try:
                d = json.loads(ln); done.add((d['pkg'], d['name']))
            except Exception:
                pass
    t0 = time.time()
    with JSONL.open('a', encoding='utf-8') as fo:
        for pkg, d, _ in DIRS:
            D = Path(d)
            if not D.exists():
                print(f'[skip] {pkg}'); continue
            for i, n in enumerate(names, 1):
                if (pkg, n) in done:
                    continue
                if free_gb() < 3:
                    print('!! 内存不足, 停止'); return report()
                p = D / f'{n}.jpg'
                if not p.exists():
                    p = D / f'{n}.png'
                if not p.exists():
                    continue
                im = Image.open(p).convert('RGB')
                arr = np.asarray(im)
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
                rec = {'pkg': pkg, 'name': n}
                for k, m in ms.items():
                    with torch.no_grad():
                        rec[k] = round(float(m(t).item()), 5)
                fo.write(json.dumps(rec) + '\n'); fo.flush()
                del im, arr, t
                if i % 10 == 0:
                    gc.collect(); torch.cuda.empty_cache()
            print(f'  [{pkg}] 完成 ({time.time()-t0:.0f}s, 可用 {free_gb():.1f} GB)', flush=True)
    report()


def report():
    import numpy as np
    if not JSONL.exists():
        print('无结果'); return
    recs = [json.loads(x) for x in JSONL.read_text(encoding='utf-8').splitlines() if x.strip()]
    by = {}
    for r in recs:
        by.setdefault(r['pkg'], {})[r['name']] = r
    # 只用四包共同覆盖、且条数相同的子集
    common = None
    for pkg in by:
        s = set(by[pkg])
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if not common:
        print('无共同子集'); return
    res = {pkg: {k: round(float(np.mean([by[pkg][n][k] for n in common])), 4) for k in KEYS}
           for pkg in by}
    scores = {p: sc for p, _, sc in DIRS if sc is not None}
    print(f'\n=== 四包本地指标 (测试集 {len(common)} 张同名配对) ===')
    print(f'  {"包":<8}{"官方分":>9}' + ''.join(f'{k:>10}' for k in KEYS))
    for pkg, _, sc in DIRS:
        if pkg in res:
            row = f'  {pkg:<8}{(f"{sc:.2f}" if sc else "待出"):>9}'
            row += ''.join(f'{res[pkg][k]:>10.4f}' for k in KEYS)
            print(row)
    # 回归: 用有分的包
    pts = [(p, sc) for p, _, sc in DIRS if sc is not None and p in res]
    if len(pts) >= 2:
        y = np.array([s for _, s in pts])
        print(f'\n=== 单指标线性拟合 (n={len(pts)}, 分数跨度 {y.max()-y.min():.1f}) ===')
        for k in KEYS:
            x = np.array([res[p][k] for p, _ in pts])
            b, a0 = np.polyfit(x, y, 1)
            pred = a0 + b * x
            r = y - pred
            rmse = float(np.sqrt((r ** 2).mean()))
            flag = '  <== 最优' if False else ''
            print(f'  {k:<9} Score = {a0:8.3f} {b:+9.3f}*{k}   rmse={rmse:6.3f}'
                  f'   残差 ' + ' '.join(f'{p}={v:+.1f}' for (p, _), v in zip(pts, r)))
    OUT.write_text(json.dumps({'n': len(common), 'metrics': res,
                               'scores': scores}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
