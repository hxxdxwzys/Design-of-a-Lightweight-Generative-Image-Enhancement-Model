r"""单次时延测量助手：python opt11_lat1.py <model_dir> [n]"""
import importlib.util, os, statistics, sys, time
import torch
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
import score01_bench as SB
md = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
x = SB.load_input("c1_512")
spec = importlib.util.spec_from_file_location("lh", os.path.join(md, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(md)
with torch.no_grad():
    for _ in range(2): R.infer(x, prompt="")
    ts = []
    for _ in range(n):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
s = sorted(ts)
print("LAT median=%.2f min=%.2f p90=%.2f max=%.2f" % (statistics.median(s), s[0], s[min(n-1,int(0.9*n))], s[-1]), flush=True)
