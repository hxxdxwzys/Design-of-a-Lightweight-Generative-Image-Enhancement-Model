r"""同一 V4 包（学生+thin2 refiner+unsharp）：benchmark=False vs True 的同进程交替对照。"""
import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_TinyRefV4\model_dir"

def build(bench):
    torch.backends.cudnn.benchmark = bool(bench)
    spec = importlib.util.spec_from_file_location("rb_%d" % int(bench), os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Runner(MD)

def timeit(R, n=12):
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(n):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    return statistics.median(ts), min(ts)

t0 = time.perf_counter(); R0 = build(False); load0 = time.perf_counter() - t0
t0 = time.perf_counter(); R1 = build(True);  load1 = time.perf_counter() - t0
print("[load] benchmark=False %.1f s | True %.1f s" % (load0, load1), flush=True)
res = {0: [], 1: []}
for rnd in range(4):
    for key, R in ((0, R0), (1, R1)):
        with torch.no_grad():
            ts = []
            for _ in range(10):
                t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
        res[key].extend(ts)
for key, label in ((0, "benchmark=False"), (1, "benchmark=True ")):
    v = res[key]; med = statistics.median(v); r_ = 285.6/(med+2)
    print("[%-16s] median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (label, med, min(v), r_, (r_/0.78)**0.19), flush=True)
print("BENCH_AB_DONE", flush=True)
