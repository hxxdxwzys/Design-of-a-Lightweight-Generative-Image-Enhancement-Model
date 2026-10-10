import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_probe_thin\model_dir"
r = subprocess.run([PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                    "--refiner", r"D:\models\refine_v1\refiner_thin_s8000.pt", "--modeldir", MD, "--unsharp", "60"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
print("[export]", r.returncode, (r.stdout or "").strip().splitlines()[-2:], flush=True)
spec = importlib.util.spec_from_file_location("rthin", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
print("[graph captured]", R.graph is not None, flush=True)
with torch.no_grad():
    for _ in range(3): R.infer(x, prompt="")
    ts = []
    for _ in range(20):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts); r_ = 285.6/(med+2)
print("[latency] median %.2f ms best %.2f -> r=%.1f factor=%.3f" % (med, min(ts), r_, (r_/0.78)**0.19), flush=True)
# 顺带对照: 纯学生 runner
md2 = r"D:\models\pkg_Tinyv21\model_dir"
spec2 = importlib.util.spec_from_file_location("r21", os.path.join(md2, "runner.py"))
m2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(m2)
R2 = m2.Runner(md2)
with torch.no_grad():
    for _ in range(3): R2.infer(x, prompt="")
    ts2 = []
    for _ in range(20):
        t = time.perf_counter(); R2.infer(x, prompt=""); ts2.append((time.perf_counter()-t)*1e3)
print("[latency 学生单独] median %.2f ms" % statistics.median(ts2), flush=True)
