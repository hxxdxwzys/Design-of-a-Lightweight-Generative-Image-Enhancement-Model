import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_TinyRefV6\model_dir"
spec = importlib.util.spec_from_file_location("rv6", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
with torch.no_grad():
    for _ in range(3): R.infer(x, prompt="")
    ts = []
    for _ in range(25):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
print("V6(r256c@10k): median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (med, min(ts), r_, fac))
for c in (0.78, 0.8046, 0.82):
    IT = 39.2*(c-0.2752)**0.4777
    print("   若 clipiqa=%.4f -> IT=%.2f -> 预测 %.1f" % (c, IT, IT*fac))
