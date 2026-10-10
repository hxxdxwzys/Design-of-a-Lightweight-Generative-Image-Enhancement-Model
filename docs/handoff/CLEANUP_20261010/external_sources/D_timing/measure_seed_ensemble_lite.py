# -*- coding: utf-8 -*-
"""种子平均判决 —— 内存安全版 (2026-09-12 v3)
================================================================
两次翻车的教训 (同一根因):
  · verify_probe_a_metrics.py : 4K float32 numpy 堆积       -> 私有内存 18.3 GB
  · eval.py (CPU, 原生 4K)    : LPIPS/DISTS 的 VGG 特征图  -> 私有内存 27.5 GB
  => 结论: 【原生 4K 上禁止计算 LPIPS/DISTS】(GPU 与 CPU 都不行)。
本版只算内存有界的指标:
  · niqe  (无参考, 且是我们的得分假设轴) —— 单图处理, 峰值几百 MB
  · psnr / ssim (全参考但无 VGG) —— 同样有界
  · 逐图 del + gc, 带可用内存守卫
用法: python measure_seed_ensemble_lite.py
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # 只用 CPU, 不与 GPU 渲染争抢
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
WORK = Path(r'D:/models/seed_ensemble_test')
OUT = Path(r'D:/tmp_timing/seed_ensemble_lite.json')
JSONL = Path(r'D:/tmp_timing/seed_ensemble_lite.jsonl')
MIN_FREE_GB = 4.0
REF_4K = {'niqe': 5.1723, 'psnr': 24.604, 'ssim': 0.7417}     # sc0.8 单样本 (sweep_scale_val.json, 原生 4K)


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
    assert free_gb() > MIN_FREE_GB, f'可用内存仅 {free_gb():.1f} GB, 拒绝启动'
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    ms = {k: pyiqa.create_metric(k, device='cpu') for k in ('niqe', 'psnr', 'ssim')}
    print(f'[metric] {list(ms)} (device=cpu) | 可用内存 {free_gb():.1f} GB', flush=True)

    done = set()
    if JSONL.exists():
        for ln in JSONL.read_text(encoding='utf-8').splitlines():
            try:
                d = json.loads(ln); done.add((d['tag'], d['name']))
            except Exception:
                pass

    t0 = time.time()
    with JSONL.open('a', encoding='utf-8') as fo:
        for tag in ('s8', 'avg4'):
            d = WORK / tag
            if not d.exists():
                continue
            for g in sorted(VAL.glob('*_gt.*')):
                stem = g.name.split('_gt')[0]
                p = d / f'{stem}_lq.png'
                if not p.exists() or (tag, stem) in done:
                    continue
                if free_gb() < MIN_FREE_GB:
                    print(f'!! 可用内存 {free_gb():.1f} GB 不足, 停止'); return report()
                pred = Image.open(p).convert('RGB')
                gt = Image.open(g).convert('RGB')
                ap = np.asarray(pred); ag = np.asarray(gt)
                tp = torch.from_numpy(ap).permute(2, 0, 1).unsqueeze(0).float().div_(255)
                tg = torch.from_numpy(ag).permute(2, 0, 1).unsqueeze(0).float().div_(255)
                rec = {'tag': tag, 'name': stem}
                with torch.no_grad():
                    rec['niqe'] = round(float(ms['niqe'](tp).item()), 4)
                    rec['psnr'] = round(float(ms['psnr'](tp, tg).item()), 4)
                    rec['ssim'] = round(float(ms['ssim'](tp, tg).item()), 4)
                fo.write(json.dumps(rec) + '\n'); fo.flush()
                print(f'  [{tag}] {stem}: niqe={rec["niqe"]} psnr={rec["psnr"]} ssim={rec["ssim"]} '
                      f'({time.time()-t0:.0f}s, 可用 {free_gb():.1f} GB)', flush=True)
                del pred, gt, ap, ag, tp, tg
                gc.collect()
    report()


def report():
    import numpy as np
    if not JSONL.exists():
        print('无结果'); return
    recs = [json.loads(x) for x in JSONL.read_text(encoding='utf-8').splitlines() if x.strip()]
    by = {}
    for r in recs:
        by.setdefault(r['tag'], {})[r['name']] = r
    if 's8' not in by or 'avg4' not in by:
        print('数据不全'); return
    common = sorted(set(by['s8']) & set(by['avg4']))
    res = {}
    for k in ('niqe', 'psnr', 'ssim'):
        a = float(np.mean([by['s8'][n][k] for n in common]))
        b = float(np.mean([by['avg4'][n][k] for n in common]))
        res[k] = {'single': round(a, 4), 'avg4': round(b, 4), 'delta': round(b - a, 4)}
    print(f'\n=== 4 种子平均 vs 单样本 (验证集 {len(common)} 张, 原生 4K) ===')
    print(f'  {"指标":<8}{"单样本":>10}{"4种子平均":>12}{"Δ":>10}{"sc0.8 参照":>12}')
    for k, v in res.items():
        print(f'  {k:<8}{v["single"]:>10.4f}{v["avg4"]:>12.4f}{v["delta"]:>+10.4f}{REF_4K.get(k,0):>12.4f}')
    dq = res['niqe']['delta']
    print(f'\n  判决 (门槛: niqe 至少降 0.30, 与一个 scale 档同量级):')
    print(f'    niqe Δ = {dq:+.4f} -> ' +
          ('✅ 值得花 12.7h 做全量种子平均包' if dq <= -0.30 else
           '⚠️ 增益偏小, 建议只做 N=2 试一发' if dq <= -0.10 else
           '❌ 增益可忽略, 砍掉阶段 2'))
    OUT.write_text(json.dumps({'n': len(common), 'res': res, 'ref_sc08': REF_4K},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'已存 {OUT}')


if __name__ == '__main__':
    main()
