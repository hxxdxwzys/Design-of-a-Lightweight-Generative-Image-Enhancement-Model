# -*- coding: utf-8 -*-
r"""解耦件素材实测：候选 runner 的 512 空闲时延 + 是否真在做增强（lapvar / 与输入的平均差）。
不占用 VOSR 链总账（这些不是 VOSR 资产）；读数落盘 D:\models\opt11\decoup_measure.json。
"""
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB  # noqa: E402

PKGS = ['pkg_TinyV4', 'pkg_TinyV4g', 'pkg_TinyV2g', 'pkg_TinyV7gan', 'pkg_TinyV9b', 'pkg_TinyV11b']
BASE = 285.6
x = SB.load_input('c1_512')


def lapvar(t):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3).to(t.device)
    g = F.conv2d(t.mean(1, keepdim=True), k, padding=1)
    return float(g.var() * 255.0 * 255.0)


def load_runner(model_dir):
    spec = importlib.util.spec_from_file_location('cand_runner', str(Path(model_dir) / 'runner.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(str(model_dir))


out = {'schema': 'decoup-measure/1', 'at': time.strftime('%Y-%m-%d %H:%M:%S'),
       'input': 'c1_512', 'lapvar_input': lapvar(x), 'baseline_ms': BASE, 'pkgs': {}}
print('输入 lapvar = %.1f' % out['lapvar_input'], flush=True)
print('%-14s %8s %8s %9s %8s %9s %9s' % ('pkg', 'load_s', 'first', 'median', 'best', 'speedup', 'lapvar'), flush=True)
for name in PKGS:
    md = os.path.join(r'D:\models', name, 'model_dir')
    if not os.path.isdir(md):
        print('%-14s MISSING' % name, flush=True)
        continue
    t0 = time.perf_counter()
    R = load_runner(md)
    load_s = time.perf_counter() - t0
    with torch.no_grad():
        t = time.perf_counter(); y = R.infer(x, prompt=''); first = (time.perf_counter() - t) * 1e3
        for _ in range(2):
            y = R.infer(x, prompt='')
        ts = []
        for _ in range(10):
            t = time.perf_counter(); y = R.infer(x, prompt=''); ts.append((time.perf_counter() - t) * 1e3)
    med, best = statistics.median(ts), min(ts)
    u8i = torch.round((x.clamp(-1, 1) + 1) * 127.5).cpu()
    u8o = torch.round((y.clamp(-1, 1) + 1) * 127.5).cpu()
    mad = float((u8o - u8i).abs().mean())
    lv = lapvar(y)
    out['pkgs'][name] = {'load_s': load_s, 'first_ms': first, 'timed_ms': ts, 'median_ms': med,
                         'best_ms': best, 'speedup': BASE / med, 'lapvar_out': lv,
                         'mean_abs_diff_u8': mad, 'shape_out': list(y.shape)}
    print('%-14s %8.2f %8.1f %9.2f %8.2f %9.1f %9.1f  (|Δu8|=%.2f)' %
          (name, load_s, first, med, best, BASE / med, lv, mad), flush=True)
    del R
    torch.cuda.empty_cache()
Path(r'D:\models\opt11').mkdir(parents=True, exist_ok=True)
Path(r'D:\models\opt11\decoup_measure.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('[done] -> D:\\models\\opt11\\decoup_measure.json', flush=True)
