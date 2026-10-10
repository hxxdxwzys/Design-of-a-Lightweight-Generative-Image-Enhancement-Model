import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
names = [r"D:\models\pkg_Tinyv20b\model_dir", r"D:\models\pkg_Tinyv21\model_dir"]
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
c = {"pkg_Tinyv20b": 0.6128, "pkg_Tinyv21": 0.6306}
for md in names:
    v = res[md]; med = statistics.median(v); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
    tag = os.path.basename(os.path.dirname(md)); cc = c[tag]
    print("%-14s median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (tag, med, min(v), r_, fac))
    for label, (A, b) in (("保守", (39.2, 0.4777)), ("乐观", (64.4, 0.816))):
        IT = A * (cc - 0.2752) ** b
        print("      clipiqa=%.4f %s映射 IT=%.2f -> 预测 %.1f" % (cc, label, IT, IT*fac))
