import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
md = r"D:\models\pkg_TinyV18\model_dir"
spec = importlib.util.spec_from_file_location("r18m", os.path.join(md, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(md)
with torch.no_grad():
    for _ in range(2): R.infer(x, prompt="")
    ts = []
    for _ in range(10):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts)
print("v18 runner 512: median %.2f ms (best %.2f) -> r = 285.6/(t+2) = %.1f -> factor (r/0.78)^0.19 = %.3f" %
      (med, min(ts), 285.6/(med+2), (285.6/(med+2)/0.78)**0.19))
for name, c in (("保守族内(39.2·(c-0.2752)^0.4777)", (39.2, 0.4777)), ("乐观跨族(64.4·(c-0.2752)^0.816)", (64.4, 0.816))):
    IT = c[0] * (0.5715 - 0.2752) ** c[1]
    print("   %s: IT = %.2f -> 预测分 %.1f" % (name, IT, IT * (285.6/(med+2)/0.78)**0.19))
