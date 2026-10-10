# -*- coding: utf-8 -*-
"""q6_export.py — 把扁平学生导出为 fp16 JIT + CUDA Graph runner, 测时延与输出真实性.
用法: python q6_export.py [--ckpt ...] [--pkg D:/models/pkg_FlatS]
"""
import os, sys, time, json, argparse, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'D:/tmp_timing')
from q6_train_flat import Flat1Step   # 复用同一架构定义

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=r'D:/models/tiny_v1/flat_student.pt')
ap.add_argument('--pkg', default=r'D:/models/pkg_FlatS')
a = ap.parse_args()
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
dev = 'cuda'
pkg = Path(a.pkg); (pkg / 'model_dir').mkdir(parents=True, exist_ok=True)

ck = torch.load(a.ckpt, map_location='cpu')
m = Flat1Step(**ck['cfg']).eval()
m.load_state_dict(ck['sd'])
print('[ckpt] cfg=%s params=%.3fM' % (ck['cfg'], sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
mh = m.half().to(dev).eval()
ex = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    traced = torch.jit.trace(mh, ex, strict=False)
traced = torch.jit.freeze(traced.eval())
pt = pkg / 'model_dir' / 'your_model.pt'
torch.jit.save(traced, str(pt))
print('[jit] saved %s  %.1f KB' % (pt, pt.stat().st_size / 1024), flush=True)

# ---- CUDA Graph + pinned staging ----
mod = torch.jit.load(str(pt), map_location=dev).eval()
static_in = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
cpu_stage = torch.empty(1, 3, 512, 512, dtype=torch.float16, pin_memory=True)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): mod(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = mod(static_in)
x_cpu = torch.randn(1, 3, 512, 512)
def infer(x):
    cpu_stage.copy_(x)
    static_in.copy_(cpu_stage, non_blocking=True)
    g.replay()
    return static_out.float().clamp(-1.0, 1.0)
def bench(fn, n=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    arr = np.array(ts)
    return float(np.median(arr)), float(arr.mean()), float(arr.min())
med, mean, mn = bench(lambda: infer(x_cpu))
print('[latency] infer(cpu in) med=%.3f mean=%.3f min=%.3f ms   r_official=%.1f  预测分=%.2f' % (
    med, mean, mn, 285.6 / (med + 2.515),
    43.429 * ((285.6 / (med + 2.515)) / 0.78) ** 0.198), flush=True)
med2 = bench(lambda: g.replay())[0]
print('[latency] graph.replay med=%.3f ms (模型本体)' % med2, flush=True)

# ---- 输出真实性门禁 (在官方最可能的 512 输入口径上) ----
def load512(nm):
    lq = Image.open(TEST / (nm + '.jpg')).convert('RGB').resize((512, 512), Image.BICUBIC)
    m24 = Image.open(M24D / (nm + '.jpg')).convert('RGB').resize((512, 512), Image.BICUBIC)
    f = lambda im: torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
    return f(lq), f(m24)
def lv(t):
    ggt = (t.float().mean(1, keepdim=True) * 255)
    lap = (-4 * ggt[..., 1:-1, 1:-1] + ggt[..., :-2, 1:-1] + ggt[..., 2:, 1:-1] + ggt[..., 1:-1, :-2] + ggt[..., 1:-1, 2:])
    return float(lap.var().item())
rows = []
for nm in ['case1', 'case12', 'case30', 'case54', 'case77', 'case91']:
    lq, m24 = load512(nm)
    o = infer(lq.unsqueeze(0)).cpu()
    rows.append((nm, float((o[0] - lq).abs().mean() * 255 / 2), float((o[0] - m24).abs().mean() * 255 / 2),
                 float((lq - m24).abs().mean() * 255 / 2), lv(lq), lv(o[0]), lv(m24),
                 float(o.abs().max()), int(torch.isnan(o).sum())))
print('%-7s %9s %9s %9s | %10s %10s %10s | %6s %4s' % ('case', '|out-LQ|', '|out-M24|', '|LQ-M24|', 'lapvar LQ', 'lapvar out', 'lapvar M24', 'max|o|', 'NaN'))
for r in rows:
    print('%-7s %9.2f %9.2f %9.2f | %10.1f %10.1f %10.1f | %6.3f %4d' % r)
shutil.copy(r'D:/tmp_timing/q6_runner_template.py', str(pkg / 'model_dir' / 'runner.py'))
print('[pkg]', pkg / 'model_dir', flush=True)
print('Q6_EXPORT_DONE')
