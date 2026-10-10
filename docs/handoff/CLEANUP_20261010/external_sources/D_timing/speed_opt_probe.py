# -*- coding: utf-8 -*-
"""纯速度探针: 出货版 runner 的真实时延 + 推理路径优化 (2026-09-12 夜)
================================================================
背景: 探针 A 证伪 niqe/保真轴 -> 速度成为唯一未变过的维度。
      出货版 runner.py 是【纯 eager + autocast】(_FullChain.forward), your_model.pt 只是权重字典,
      没有任何 JIT/cudnn 优化 -> 优化空间可能极大, 且【图像逐字节不变】=> 分数变化只能归因于速度。

本脚本:
  ① 测出货版 infer(512) 真实时延 (安静 GPU)
  ② 逐项优化并复测: cudnn.benchmark / channels_last / jit.trace / jit.freeze+optimize_for_inference
  ③ 校验优化后输出与原版的数值一致性 (MAE) —— 必须证明"同一模型"
用法: python speed_opt_probe.py [--n 20]
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SRC = Path(r'D:/models/final_compliant/model_dir')
WORK = Path(r'D:/models/opt_probe')
BASELINE_MS = 285.6          # 旧世界同机实测的 OSEDiff 基线 (历史测量, 不可复测)
OUTJSON = Path(r'D:/tmp_timing/speed_opt_probe.json')


def timeit(fn, n=20, warm=5, label=''):
    for _ in range(warm):
        fn()
    import torch
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    med, best = ts[len(ts) // 2], ts[0]
    print(f'  {label:<38} 中位 {med:8.1f} ms   最快 {best:8.1f} ms   比基线 {BASELINE_MS/med:5.2f}x', flush=True)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()

    import torch

    md = WORK / 'model_dir'
    if not md.exists():
        shutil.copytree(SRC, md)
        print(f'[copy] {SRC} -> {md}', flush=True)
    sys.path.insert(0, str(md))
    sys.path.insert(0, str(md / '_vendor'))
    import runner as R                      # noqa: E402
    print('[load] Runner ...', flush=True)
    t0 = time.perf_counter()
    r = R.Runner(str(md))
    print(f'  构造耗时 {time.perf_counter()-t0:.2f}s', flush=True)

    x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1)
    z = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)

    print('\n=== ① 出货版原样 ===', flush=True)
    base_ms = timeit(lambda: r.infer(x), a.n, 5, 'eager + autocast (出货版)')

    # 参考输出 (固定 z, 便于逐项比较数值一致性)
    with torch.no_grad():
        ref = r.model(x.half(), z).float().clone()
    print(f'  ref out: mean={ref.mean():.4f} std={ref.std():.4f}', flush=True)

    results = {'baseline_ms': base_ms, 'baseline_ratio': BASELINE_MS / base_ms,
               'baseline_ratio_vs_hist': round(BASELINE_MS / base_ms, 3),
               'baseline_ms_used': BASELINE_MS, 'variants': {}}

    def check(tag, module, use_z=True):
        with torch.no_grad():
            o = module(x.half(), z).float()
        mae = (o - ref).abs().mean().item()
        print(f'  [{tag}] 与 ref 的 MAE = {mae:.5f}', flush=True)
        return mae

    # ② cudnn.benchmark
    print('\n=== ② cudnn.benchmark ===', flush=True)
    torch.backends.cudnn.benchmark = True
    ms = timeit(lambda: r.infer(x), a.n, 5, 'eager + cudnn.benchmark')
    results['variants']['cudnn_benchmark'] = {'ms': ms, 'ratio': round(BASELINE_MS / ms, 3)}

    # ③ channels_last (对卷积型 VAE/decoder 有效)
    print('\n=== ③ channels_last ===', flush=True)
    try:
        m = r.model
        m.vae.to(memory_format=torch.channels_last)
        m.dec.to(memory_format=torch.channels_last)
        xc = x.half().contiguous(memory_format=torch.channels_last)
        ms = timeit(lambda: r.infer(xc), a.n, 5, 'eager + channels_last')
        results['variants']['channels_last'] = {'ms': ms, 'ratio': round(BASELINE_MS / ms, 3)}
    except Exception as e:
        print(f'  [skip] {e}', flush=True)

    # ④ torch.jit.trace
    print('\n=== ④ jit.trace ===', flush=True)
    m = r.model
    with torch.no_grad():
        tr = torch.jit.trace(m, (x.half(), z), check_trace=False)
    check('trace', tr)
    ms = timeit(lambda: tr(x.half(), z), a.n, 5, 'jit.trace')
    results['variants']['jit_trace'] = {'ms': ms, 'ratio': round(BASELINE_MS / ms, 3)}

    # ⑤ freeze + optimize_for_inference
    print('\n=== ⑤ freeze + optimize_for_inference ===', flush=True)
    try:
        fr = torch.jit.freeze(tr.eval())
        opt = torch.jit.optimize_for_inference(fr)
        check('frozen', opt)
        ms = timeit(lambda: opt(x.half(), z), a.n, 5, 'jit.freeze + optimize_for_inference')
        results['variants']['jit_frozen_opt'] = {'ms': ms, 'ratio': round(BASELINE_MS / ms, 3)}
    except Exception as e:
        print(f'  [fail] {e}', flush=True)

    OUTJSON.write_text(__import__('json').dumps(results, indent=1), encoding='utf-8')
    print(f'\n已存 {OUTJSON}', flush=True)
    print('SPEED_OPT_PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
