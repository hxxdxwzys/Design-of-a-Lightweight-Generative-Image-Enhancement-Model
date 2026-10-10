import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_TinyRefV4\model_dir"
spec = importlib.util.spec_from_file_location("rv4", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
with torch.no_grad():
    for _ in range(3): R.infer(x, prompt="")
    ts = []
    for _ in range(20):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
IT = 39.2 * (0.8046 - 0.2752) ** 0.4777
print("V4 包: graph=%s median=%.2f ms best=%.2f -> r=%.1f factor=%.3f | IT=%.2f -> 预测(保守)=%.1f" %
      (R.graph is not None, med, min(ts), r_, fac, IT, IT*fac))
