import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
names = [r"D:\models\pkg_Tinyv21\model_dir", r"D:\models\pkg_TinyRefV2\model_dir"]
Rs = {}
for md in names:
    spec = importlib.util.spec_from_file_location("m_" + os.path.basename(os.path.dirname(md)), os.path.join(md, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    Rs[md] = m.Runner(md)
res = {md: [] for md in names}
with torch.no_grad():
    for md in names:
        for _ in range(2): Rs[md].infer(x, prompt="")
    for rnd in range(4):
        for md in names:
            for _ in range(10):
                t = time.perf_counter(); Rs[md].infer(x, prompt=""); res[md].append((time.perf_counter()-t)*1e3)
for md in names:
    v = res[md]; med = statistics.median(v); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
    print("%-16s median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (os.path.basename(os.path.dirname(md)), med, min(v), r_, fac))
