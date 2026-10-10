# -*- coding: utf-8 -*-
"""测试集 30 张 niqe/clipiqa 快测 (2026-09-12) —— 省内存版, 逐图追加落盘
用途: 对任意已渲好的 output_dir, 测 sorted(测试集)[:30] 前缀的 niqe+clipiqa,
      与 M24(sc0.8) 同 30 张配对, 并用 niqe 线性模型给出分数预测。
用法: python measure_test30_niqe.py --dir <output_dir> --tag <名字>
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
M24 = Path(r'D:/models/final_compliant/output_dir')
M24_30 = {'niqe': 5.2559, 'clipiqa': 0.9023}
REF_SCORE = 43.49          # 模型对 M24 的点预测 (官方实测 43.6951)


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
    ap.add_argument('--dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    D = Path(a.dir)
    JL = Path(rf'D:/tmp_timing/test30_{a.tag}.jsonl')
    OUT = Path(rf'D:/tmp_timing/test30_{a.tag}.json')

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ms = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'clipiqa')}
    names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
    print(f'[{a.tag}] {len(names)} 张 | {list(ms)} on {dev} | 可用内存 {free_gb():.1f} GB', flush=True)

    vals = {'niqe': [], 'clipiqa': []}
    t0 = time.time()
    with JL.open('a', encoding='utf-8') as fo:
        for i, n in enumerate(names, 1):
            p = D / f'{n}.jpg'
            if not p.exists():
                p = D / f'{n}.png'
            if not p.exists():
                continue
            if free_gb() < 3:
                print('!! 内存不足, 停止'); break
            im = Image.open(p).convert('RGB')
            arr = np.asarray(im)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            rec = {'name': n}
            for k, m in ms.items():
                with torch.no_grad():
                    rec[k] = round(float(m(t).item()), 5)
                vals[k].append(rec[k])
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            del im, arr, t
            if i % 10 == 0:
                gc.collect(); torch.cuda.empty_cache()
                print(f'  {i}/{len(names)} ({time.time()-t0:.0f}s)', flush=True)

    res = {k: round(float(np.mean(v)), 4) for k, v in vals.items() if v}
    print(f'\n=== {a.tag} vs M24 (测试集 {len(vals["niqe"])} 张同口径) ===')
    print(f'  {"指标":<9}{"M24(sc0.8)":>12}{a.tag:>12}{"Δ":>10}')
    for k in ('niqe', 'clipiqa'):
        if k in res:
            print(f'  {k:<9}{M24_30[k]:>12.4f}{res[k]:>12.4f}{res[k]-M24_30[k]:>+10.4f}')
    pred = REF_SCORE - 11.681 * (res.get('niqe', M24_30['niqe']) - M24_30['niqe'])
    print(f'\n  niqe 线性模型预测: M24={REF_SCORE:.2f} -> {a.tag}={pred:.2f} (Δ={pred-REF_SCORE:+.2f})')
    print(f'  （参照: sc0.4 实测 test niqe 4.0976 -> 预测 57.02）')
    OUT.write_text(json.dumps({'tag': a.tag, 'dir': str(D), 'n': len(vals['niqe']),
                               'metrics': res, 'm24_30': M24_30, 'pred': round(pred, 2)},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'已存 {OUT}')


if __name__ == '__main__':
    main()
