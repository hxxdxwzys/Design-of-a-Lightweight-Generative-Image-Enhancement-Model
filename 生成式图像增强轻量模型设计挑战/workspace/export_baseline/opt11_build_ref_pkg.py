r"""一条命令出包: 合并导出(学生+refiner) -> 用打包 runner 渲染 100 张 4K -> 打 zip。
用法: python opt11_build_ref_pkg.py <tag> <student_ckpt> <refiner_ckpt>
"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
tag, ckpt, refiner = sys.argv[1], sys.argv[2], sys.argv[3]
PKG = Path(r"D:\models") / ("pkg_" + tag)
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")

def R(name, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    print("[%s] rc=%d %.1fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)), flush=True)
    if r.returncode != 0:
        print("   stderr:", (r.stderr or "")[-500:], flush=True)
    return r.returncode

R("export_combined", [PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", ckpt, "--refiner", refiner, "--modeldir", str(PKG / "model_dir")], 1800)
R("render100_pkg", [PY, "-u", str(HERE / "opt11_render_pkg.py"), "--modeldir", str(PKG / "model_dir"), "--out", str(PKG / "output_dir")], 3600)
n = len(list((PKG / "output_dir").glob("*.jpg")))
print("[render] jpg =", n, flush=True)
R("package", [PY, str(T / "package_submit_v2.py"), "--out-dir", str(PKG / "output_dir"), "--model-dir", str(PKG / "model_dir"),
              "--name", tag, "--zip", str(Path(r"D:\models\opt11") / ("Track2_%s.zip" % tag))], 1800)
print("BUILD_REF_PKG_DONE", tag, flush=True)
