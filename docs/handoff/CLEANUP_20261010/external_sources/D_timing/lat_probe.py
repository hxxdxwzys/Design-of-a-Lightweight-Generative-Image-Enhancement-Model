# -*- coding: utf-8 -*-
"""时延协议诊断：比较 (a) 无 sync 单次计时 (b) 前后 sync (c) 长期循环均摊。
用法: python lat_probe.py <model_dir>
"""
import importlib.util, os, statistics, sys, time
import torch

d = sys.argv[1]
spec = importlib.util.spec_from_file_location("rr", os.path.join(d, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(d)
print("[info] graph=%s device=%s" % (R.graph is not None, R.device))
if torch.cuda.is_available():
    print("[info] device name=%s" % torch.cuda.get_device_name(0))

x = (torch.rand(1, 3, 512, 512, device="cuda") * 2 - 1).contiguous()

# preheat 30 s
t0 = time.time(); i = 0
with torch.no_grad():
    while time.time() - t0 < 30.0:
        R.infer(x, prompt=""); i += 1
torch.cuda.synchronize()
print("[preheat] %.0f s / %d iters -> %.2f ms/iter (synced at end)" % (time.time() - t0, i, (time.time() - t0) / i * 1e3))

# (a) naive: no sync (what infer.py --timing and selfcheck.py do)
ts = []
with torch.no_grad():
    for _ in range(50):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter() - t) * 1e3)
s = sorted(ts)
print("(a) no-sync per-call      median=%7.2f  min=%7.2f  p90=%7.2f" % (statistics.median(s), s[0], s[int(0.9*len(s))]))

# (b) sync around each call
ts = []
with torch.no_grad():
    for _ in range(50):
        torch.cuda.synchronize(); t = time.perf_counter()
        R.infer(x, prompt=""); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
s = sorted(ts)
print("(b) sync-both per-call    median=%7.2f  min=%7.2f  p90=%7.2f" % (statistics.median(s), s[0], s[int(0.9*len(s))]))

# (c) amortized: 200 calls inside one timed region, sync once at end
for n in (100, 200):
    with torch.no_grad():
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            R.infer(x, prompt="")
        torch.cuda.synchronize()
        el = (time.perf_counter() - t) * 1e3
    print("(c) amortized %3d calls   per-call=%7.3f ms  (total %.1f ms)" % (n, el / n, el))

# (d) 4K render throughput reference (throughput, single image)
from PIL import Image
import numpy as np
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
sys.path.insert(0, r"D:\models\finals_20260927\Track2_light_diffusion_inference")
import infer as I
lq = np.asarray(Image.open(os.path.join(TEST, "case3.jpg")).convert("RGB"), dtype=np.uint8)
t = time.perf_counter()
out = I.render_4k(R, lq, I.make_feather(I.TILE, I.OVER), str(R.device))
torch.cuda.synchronize()
el = time.perf_counter() - t
Ht, Wt = lq.shape[:2]
ntile = len(range(0, Ht - I.TILE + 1, I.TILE - I.OVER)) + 1
ntilex = len(range(0, Wt - I.TILE + 1, I.TILE - I.OVER)) + 1
print("(d) 4K render wall=%.2f s  tiles=%dx%d=%d  -> %.1f ms/tile" % (el, ntile, ntilex, ntile*ntilex, el/ (ntile*ntilex) *1e3))
