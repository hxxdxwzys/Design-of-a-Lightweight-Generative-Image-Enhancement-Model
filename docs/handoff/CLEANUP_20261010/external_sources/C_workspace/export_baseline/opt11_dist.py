r"""V4 包 100 次调用的时延分布 + GPU 时钟/功耗观测（找是否存在"快路径"/双峰）。"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_TinyRefV4\model_dir"
def clocks():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        return o
    except Exception as e:
        return str(e)
spec = importlib.util.spec_from_file_location("rv4d", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
print("[clocks before]", clocks(), flush=True)
with torch.no_grad():
    first = []
    for _ in range(5):
        t = time.perf_counter(); R.infer(x, prompt=""); first.append((time.perf_counter()-t)*1e3)
    print("[first 5 calls] %s" % ["%.2f" % v for v in first], flush=True)
    ts = []
    t0 = time.perf_counter()
    for i in range(100):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    total = time.perf_counter() - t0
print("[clocks after ]", clocks(), flush=True)
s = sorted(ts)
def pct(p): return s[min(len(s)-1, int(p*len(s)))]
print("[100 calls] min %.2f p10 %.2f p25 %.2f median %.2f p75 %.2f p90 %.2f max %.2f mean %.2f" %
      (s[0], pct(0.10), pct(0.25), statistics.median(s), pct(0.75), pct(0.90), s[-1], statistics.mean(s)), flush=True)
print("[直方图] <3ms=%d  3-4=%d  4-5=%d  5-6=%d  6-7=%d  7-8=%d  >8=%d  (wall %.1f ms/call)" %
      (sum(1 for v in ts if v < 3), sum(1 for v in ts if 3 <= v < 4), sum(1 for v in ts if 4 <= v < 5),
       sum(1 for v in ts if 5 <= v < 6), sum(1 for v in ts if 6 <= v < 7), sum(1 for v in ts if 7 <= v < 8),
       sum(1 for v in ts if v >= 8), total/100*1000), flush=True)
print("DIST_DONE", flush=True)
