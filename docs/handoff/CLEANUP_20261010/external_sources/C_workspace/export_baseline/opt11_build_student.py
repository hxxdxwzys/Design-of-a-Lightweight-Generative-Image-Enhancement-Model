# -*- coding: utf-8 -*-
"""学生包构建通用脚本: python opt11_build_student.py <tag> <ckpt>
   导出 model_dir -> 渲染 100 张 4K -> 打包 zip；随后打印 512 空闲时延与预测分。"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
tag = sys.argv[1]; ckpt = sys.argv[2]
PKG = Path(r"D:\models") / ("pkg_Tiny" + tag)
(PKG / "model_dir").mkdir(parents=True, exist_ok=True)
(PKG / "output_dir").mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")

def R(name, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    print("[%s] rc=%d %.1fs :: %s" % (name, r.returncode, time.time() - t0, " | ".join(tail)), flush=True)
    if r.returncode != 0:
        print("   stderr:", (r.stderr or "")[-400:], flush=True)
    return r.returncode

R("export", [PY, "-u", str(T / "export_tiny.py"), "--ckpt", ckpt, "--modeldir", str(PKG / "model_dir"), "--down", "2"], 1200)
R("render100", [PY, "-u", str(T / "render_tiny.py"), "--ckpt", ckpt, "--down", "2", "--out", str(PKG / "output_dir")], 3600)
print("[render] jpg =", len(list((PKG / "output_dir").glob("*.jpg"))), flush=True)
R("package", [PY, str(T / "package_submit_v2.py"), "--out-dir", str(PKG / "output_dir"),
              "--model-dir", str(PKG / "model_dir"), "--name", "Tiny" + tag,
              "--zip", str(Path(r"D:\models\opt11") / ("Track2_Tiny%s.zip" % tag))], 1800)

# 空闲态 512 时延
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
import score01_bench as SB
x = SB.load_input("c1_512")
spec = importlib.util.spec_from_file_location("rm", str(PKG / "model_dir" / "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Rr = m.Runner(str(PKG / "model_dir"))
with torch.no_grad():
    for _ in range(2): Rr.infer(x, prompt="")
    ts = []
    for _ in range(10):
        t = time.perf_counter(); Rr.infer(x, prompt=""); ts.append((time.perf_counter() - t) * 1e3)
med = statistics.median(ts)
r_ = 285.6 / (med + 2); fac = (r_ / 0.78) ** 0.19
print("[latency] 空闲态 median %.2f ms (best %.2f) -> r=%.1f factor=%.3f" % (med, min(ts), r_, fac), flush=True)
