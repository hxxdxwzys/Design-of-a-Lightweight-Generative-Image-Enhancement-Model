# -*- coding: utf-8 -*-
"""探针 A 指标复核 v2 —— 省内存 / 崩溃安全版 (2026-09-12)
================================================================
v1 的教训:
  · 把 4K 图读成 float32 numpy (4096x3072x3x4 = 151 MB/张), 循环里几十张不释放
    -> 私有内存涨到 18.3 GB, 把整机 31.7 GB 物理内存打到 97.8%, 被强制终止。
  · 只在最后写一次 JSON -> 杀进程后 25/30 的结果全部丢失。
本版修正:
  · 只用 uint8 numpy (37 MB/张) -> 立刻搬到 GPU 再转 float, 每轮 del + gc
  · 逐图【追加】写 JSONL -> 任何时刻中断都不丢已算好的数据
  · 只算 niqe + clipiqa (niqe 是假设轴; musiq 是 v1 的耗时与内存大头, 砍掉)
  · 内存守卫: 可用物理内存 < 3 GB 立即停止
用法: python verify_probe_a_frugal.py [--n 30]
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
SC04 = Path(r'D:/models/final_compliant_probeA_sc04/output_dir')
M24 = Path(r'D:/models/final_compliant/output_dir')
JSONL = Path(r'D:/tmp_timing/probe_a_verify.jsonl')
OUT = Path(r'D:/tmp_timing/probe_a_verify.json')
MIN_FREE_GB = 3.0


class MEMSTATUS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb():
    st = MEMSTATUS()
    st.dwLength = ctypes.sizeof(MEMSTATUS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()

    done = set()
    if JSONL.exists():
        for ln in JSONL.read_text(encoding='utf-8').splitlines():
            try:
                d = json.loads(ln)
                done.add((d['name'], d['tag']))
            except Exception:
                pass
    print(f'[resume] 已有 {len(done)} 条结果', flush=True)

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
    ms = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'clipiqa')}
    print(f'[metric] {list(ms)} ready (device={dev}) | 可用内存 {free_gb():.1f} GB', flush=True)

    t0 = time.time()
    with JSONL.open('a', encoding='utf-8') as fo:
        for i, n in enumerate(names, 1):
            for tag, d in (('sc04', SC04), ('m24', M24)):
                if (n, tag) in done:
                    continue
                if free_gb() < MIN_FREE_GB:
                    print(f'!! 可用内存 {free_gb():.1f} GB < {MIN_FREE_GB} GB, 主动停止 (已算的都在 jsonl)', flush=True)
                    return summarize()
                p = d / f'{n}.jpg'
                if not p.exists():
                    continue
                im = Image.open(p).convert('RGB')
                arr = np.asarray(im)                       # uint8 37MB, 不转 float32
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
                rec = {'name': n, 'tag': tag}
                for k, m in ms.items():
                    with torch.no_grad():
                        rec[k] = round(float(m(t).item()), 5)
                fo.write(json.dumps(rec) + '\n')
                fo.flush()
                del im, arr, t
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f'  {i}/{a.n} ({time.time()-t0:.0f}s, 可用内存 {free_gb():.1f} GB)', flush=True)
    summarize()


def summarize():
    import numpy as np
    if not JSONL.exists():
        print('无结果'); return
    recs = [json.loads(x) for x in JSONL.read_text(encoding='utf-8').splitlines() if x.strip()]
    by = {}
    for r in recs:
        by.setdefault(r['tag'], {}).setdefault(r['name'], r)
    common = sorted(set(by.get('sc04', {})) & set(by.get('m24', {})))
    if not common:
        print('无配对结果'); return
    res = {}
    for k in ('niqe', 'clipiqa'):
        s0 = float(np.mean([by['m24'][n][k] for n in common]))
        s1 = float(np.mean([by['sc04'][n][k] for n in common]))
        res[k] = {'m24': round(s0, 4), 'sc04': round(s1, 4), 'delta': round(s1 - s0, 4)}
    print(f'\n=== 测试集 {len(common)} 张配对复核 (同 q95 jpg, 逐图同名) ===')
    print(f'  {"指标":<9}{"M24(sc0.8)":>12}{"sc0.4":>10}{"Δ":>10}')
    for k, v in res.items():
        print(f'  {k:<9}{v["m24"]:>12.4f}{v["sc04"]:>10.4f}{v["delta"]:>+10.4f}')
    nq = res['niqe']
    p1 = 104.884 - 11.681 * nq['sc04']
    p0 = 104.884 - 11.681 * nq['m24']
    print(f'\n  niqe 线性模型: M24={p0:.2f} -> sc0.4={p1:.2f}  (Δ={p1-p0:+.2f})')
    print(f'  （预登记: H1 niqe家族 50.5~58.0; 官方 M24 实测 43.6951）')
    OUT.write_text(json.dumps({'n': len(common), 'names': common, 'res': res,
                               'pred': {'m24': round(p0, 2), 'sc04': round(p1, 2)}},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'已存 {OUT}')
    print('VERIFY_FRUGAL_DONE')


if __name__ == '__main__':
    main()
