import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
r = subprocess.run([PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                    "--refiner", r"D:\models\refine_v1\refiner_gan1_s10000.pt", "--modeldir", r"D:\models\pkg_TinyRefV2\model_dir"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
print("[re-export]", r.returncode, (r.stdout or "").strip().splitlines()[-2:], flush=True)
import score01_bench as SB
x = SB.load_input("c1_512")
spec = importlib.util.spec_from_file_location("rv2", r"D:\models\pkg_TinyRefV2\model_dir\runner.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(r"D:\models\pkg_TinyRefV2\model_dir")
print("[graph captured]", R.graph is not None, flush=True)
with torch.no_grad():
    for _ in range(3): R.infer(x, prompt="")
    ts = []
    for _ in range(20):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts); r_ = 285.6/(med+2)
print("[latency 独立进程] median %.2f ms best %.2f worst %.2f -> r=%.1f factor=%.3f" % (med, min(ts), max(ts), r_, (r_/0.78)**0.19), flush=True)
