# -*- coding: utf-8 -*-
"""实测三个包的真实 512 时延 (2026-09-13) —— 把"估算的加速比"变成"实测的加速比"
必须等渲染完全停下再跑 (安静 GPU), 否则时延无意义。
用法: python measure_pkg_latency.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ENGINE'] = '1'
os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASELINE_MS = 285.6
PKGS = [
    ('M24(原版 sc0.8)', r'D:/models/final_compliant/model_dir'),
    ('M24p4(DiT 4/28)', r'D:/models/final_compliant_M24p4/model_dir'),
    ('M24avg2(2样本)', r'D:/models/final_compliant_M24avg2/model_dir'),
]
OUT = Path(r'D:/tmp_timing/pkg_latency.json')
LOG = Path(r'D:/tmp_timing/pkg_latency.log')


def main():
    import torch
    res = {}
    for name, md in PKGS:
        md = Path(md)
        if not (md / 'your_model.pt').exists():
            print(f'[skip] {name}: 包不存在'); continue
        sys.path.insert(0, str(md)); sys.path.insert(0, str(md / '_vendor'))
        for m in [k for k in list(sys.modules) if k == 'runner']:
            del sys.modules[m]
        import importlib
        runner = importlib.import_module('runner')
        try:
            r = runner.Runner(str(md))
        except Exception as e:
            print(f'[fail] {name}: {e}'); continue
        x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1)
        for _ in range(5):
            r.infer(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); r.infer(x); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        med, best = ts[len(ts) // 2], ts[0]
        res[name] = {'median_ms': round(med, 1), 'best_ms': round(best, 1),
                     'speedup': round(BASELINE_MS / med, 3)}
        line = (f'{name:<18} 中位 {med:7.1f} ms  最快 {best:7.1f} ms  '
                f'-> 加速比 {BASELINE_MS/med:.3f}x')
        print(line, flush=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
        del r
        torch.cuda.empty_cache()
        sys.path[:] = [p for p in sys.path if not str(p).startswith(str(md.parent))]
    OUT.write_text(json.dumps({'baseline_ms': BASELINE_MS, 'pkgs': res}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print(f'\n已存 {OUT}')
    print('PKG_LATENCY_DONE')


if __name__ == '__main__':
    main()
