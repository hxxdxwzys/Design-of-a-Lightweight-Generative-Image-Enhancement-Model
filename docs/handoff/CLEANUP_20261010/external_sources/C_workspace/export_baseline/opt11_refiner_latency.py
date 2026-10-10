import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
md = r"D:\models\pkg_Tinyv21\model_dir"
spec = importlib.util.spec_from_file_location("rm", os.path.join(md, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(md)
dev = R.device

class Refiner(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return (x + self.net(x)).clamp(-1, 1)

def timeit(fn, n=15, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e3)
    return statistics.median(ts)

print("== 各规格 refiner 的 512 域单次耗时 ==")
for ch, blocks in ((48, 3), (32, 2), (24, 2), (16, 2)):
    rd = torch.load(r"D:\models\refine_v1\refiner_pilot_s3000.pt", map_location="cpu", weights_only=False) if (ch, blocks) == (48, 3) else None
    ref = Refiner(ch, blocks).to(dev).eval().half()
    if rd is not None:
        ref.load_state_dict(rd["net"])
    y = torch.randn(1, 3, 512, 512, device=dev).half()
    with torch.no_grad():
        dt = timeit(lambda: ref(y))
    print("   ch=%2d blocks=%d  refiner %.2f ms" % (ch, blocks, dt))
    del ref; torch.cuda.empty_cache()

print()
print("== 端到端（同进程交替 3 轮 × 10）==")
rd = torch.load(r"D:\models\refine_v1\refiner_pilot_s3000.pt", map_location="cpu", weights_only=False)
ref = Refiner(rd["ch"], rd["blocks"]).to(dev).eval().half(); ref.load_state_dict(rd["net"])
def full_plain():
    return R.infer(x, prompt="")
def full_ref():
    y = R.infer(x, prompt="")
    with torch.no_grad():
        return ref(y.half())
a, b = [], []
with torch.no_grad():
    for _ in range(2):
        full_plain(); full_ref()
    for rnd in range(3):
        for _ in range(10):
            t0 = time.perf_counter(); full_plain(); a.append((time.perf_counter()-t0)*1e3)
        for _ in range(10):
            t0 = time.perf_counter(); full_ref(); b.append((time.perf_counter()-t0)*1e3)
for name, v in (("学生单独", a), ("学生+refiner(48/3)", b)):
    med = statistics.median(v); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
    print("   %-20s median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (name, med, min(v), r_, fac))
