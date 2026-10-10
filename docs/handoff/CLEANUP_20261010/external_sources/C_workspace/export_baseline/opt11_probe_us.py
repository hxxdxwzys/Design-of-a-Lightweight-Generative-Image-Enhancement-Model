import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
x = SB.load_input("c1_512")

def build_and_measure(unsharp, md):
    r = subprocess.run([PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                        "--refiner", r"D:\models\refine_v1\refiner_gan1_s10000.pt", "--modeldir", md, "--unsharp", str(unsharp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    if r.returncode != 0:
        print("[export fail]", unsharp, (r.stderr or "")[-300:], flush=True); return None
    spec = importlib.util.spec_from_file_location("rr_%d" % unsharp, os.path.join(md, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(md)
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(15):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    med = statistics.median(ts); r_ = 285.6/(med+2)
    print("[unsharp=%3d] graph=%s median %.2f ms best %.2f -> r=%.1f factor=%.3f" %
          (unsharp, R.graph is not None, med, min(ts), r_, (r_/0.78)**0.19), flush=True)
    del R; torch.cuda.empty_cache()
    return med

build_and_measure(0, r"D:\models\pkg_probe_us0\model_dir")
build_and_measure(60, r"D:\models\pkg_probe_us60\model_dir")
