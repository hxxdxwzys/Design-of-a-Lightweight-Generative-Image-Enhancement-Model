r"""任一 refiner 的快速判定: 导出合并模型 -> 图捕获检查 -> 空闲态时延 -> 打印判定。
用法: python opt11_probe_ref2.py <tag> <refiner_ckpt> [unsharp_pct]
"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
tag = sys.argv[1]; refiner = sys.argv[2]; us = sys.argv[3] if len(sys.argv) > 3 else "60"
MD = str(Path(r"D:\models") / ("pkg_probe_" + tag) / "model_dir")
x = SB.load_input("c1_512")
r = subprocess.run([PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                    "--refiner", refiner, "--modeldir", MD, "--unsharp", us],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
print("[export] rc=%d %s" % (r.returncode, (r.stdout or "").strip().splitlines()[-2:]), flush=True)
if r.returncode != 0:
    print((r.stderr or "")[-500:]); raise SystemExit(1)
spec = importlib.util.spec_from_file_location("rp", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
with torch.no_grad():
    for _ in range(3): R.infer(x, prompt="")
    ts = []
    for _ in range(20):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
med = statistics.median(ts); r_ = 285.6/(med+2); fac = (r_/0.78)**0.19
print("[verdict] tag=%s graph=%s median=%.2f ms best=%.2f -> r=%.1f factor=%.3f" %
      (tag, R.graph is not None, med, min(ts), r_, fac), flush=True)
print("[cmd] 若质量好, 出包: python opt11_build_ref_pkg.py TinyRefV4 D:\models\tiny_v1\v21_s21000.pt %s" % refiner, flush=True)
