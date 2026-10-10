# -*- coding: utf-8 -*-
"""H512 定位检验 (2026-09-13): 官方那个"质量项"到底评的是哪个对象?
================================================================
动机: 用新机制分 + 本地指标对照, 出现一个无法解释的矛盾 ——
  · 我们把【4K 图】的 niqe 改善 22%、lpips 改善 5% -> 分数 -0.41
  · 1.4B 家族的【4K 图】clipiqa 0.86+ 并不差 -> 却低 10~13 分
  => 官方评的很可能【不是我们提交的 4K 图】, 而是 runner 在 512 上的输出
     (那是官方 harness 手里直接就有的东西)

本检验: 用【每个包自己的 runner.py】在 512x512 上真跑 infer(), 对输出测指标,
        看排序是否与官方分一致。不花提交次数。
判据(先写死):
  官方排序: M24 43.70 > probeA 43.28 > sc00 37.63 > avg2 36.64 > p4 7.18, 1.4B 30.6~34.1
  · 512 输出的指标排序与之吻合  -> H512 成立, 此后一切优化都该打在 512 输出上
  · 排序不吻合                  -> H512 死, 质量项在 4K 图上(但那样矛盾未解)
用法: python test_h512.py [--n 10]
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUTD = Path(r'D:/tmp_timing/h512_out')
OUT = Path(r'D:/tmp_timing/h512_result.json')
PKGS = [
    ('M24',    r'D:/models/final_compliant/model_dir',            43.6951),
    ('probeA', r'D:/models/final_compliant_probeA_sc04/model_dir', 43.2837),
    ('sc00',   r'D:/models/final_compliant_M24sc00/model_dir',     37.6316),
    ('avg2',   r'D:/models/final_compliant_M24avg2/model_dir',     36.6414),
    ('p4',     r'D:/models/final_compliant_M24p4/model_dir',        7.1829),
    ('m0s1',   r'D:/models/final_compliant_14b/model_dir',         34.1056),
]
KEYS = ('clipiqa', 'niqe', 'musiq')


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
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=10)
    a = ap.parse_args()
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    outs_dir = OUTD / 'imgs'; outs_dir.mkdir(parents=True, exist_ok=True)
    # 固定输入: 前 N 张测试图缩到 512x512
    inputs = []
    for f in sorted(TEST.glob('*.jpg'))[:a.n]:
        p = OUTD / 'in' / f'{f.stem}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            Image.open(f).convert('RGB').resize((512, 512), Image.BICUBIC).save(p)
        inputs.append((f.stem, p))
    print(f'[in] {len(inputs)} 张 512 输入', flush=True)

    ms = {k: pyiqa.create_metric(k, device=dev) for k in KEYS}
    res = {}
    for tag, md, score in PKGS:
        md = Path(md)
        if not (md / 'your_model.pt').exists():
            print(f'[skip] {tag}'); continue
        sys.path.insert(0, str(md)); sys.path.insert(0, str(md / '_vendor'))
        sys.modules.pop('runner', None)
        import importlib
        try:
            R = importlib.import_module('runner')
            r = R.Runner(str(md))
        except Exception as e:
            print(f'[fail] {tag}: {type(e).__name__}: {e}', flush=True); continue
        vals = {k: [] for k in KEYS}
        for stem, p in inputs:
            im = Image.open(p).convert('RGB')
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            try:
                with torch.no_grad():
                    y = r.infer(x.to('cuda'))
            except Exception as e:
                print(f'[infer-fail] {tag} {stem}: {e}', flush=True); break
            Image.fromarray(((y[0].float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()).save(
                outs_dir / f'{tag}_{stem}.png')
            t = (y.float().clamp(-1, 1) + 1) / 2      # [-1,1] -> [0,1] (pyiqa 要求)
            for k, m in ms.items():
                with torch.no_grad():
                    vals[k].append(float(m(t).item()))
            del im, x, y, t
        gc.collect(); torch.cuda.empty_cache()
        res[tag] = {k: round(float(np.mean(v)), 4) for k, v in vals.items() if v}
        res[tag]['official'] = score
        print(f'  [{tag}] 官方 {score:7.2f} | ' +
              ' '.join(f'{k}={res[tag].get(k)}' for k in KEYS), flush=True)
        del r
        sys.path[:] = [p for p in sys.path if str(p) not in (str(md), str(md / '_vendor'))]

    print('\n=== H512 判据 ===')
    order_off = [t for t, _, _ in sorted(PKGS, key=lambda x: -x[2])]
    print(f'  官方分排序: {" > ".join(order_off)}')
    for k in KEYS:
        ok = [t for t in res if k in res[t]]
        if len(ok) < 3:
            continue
        order_k = sorted(ok, key=lambda t: -res[t][k])
        print(f'  512 输出 {k:8s} 排序: {" > ".join(order_k)}')
    OUT.write_text(json.dumps({'res': res}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}\n输出图: {outs_dir}')


if __name__ == '__main__':
    main()
