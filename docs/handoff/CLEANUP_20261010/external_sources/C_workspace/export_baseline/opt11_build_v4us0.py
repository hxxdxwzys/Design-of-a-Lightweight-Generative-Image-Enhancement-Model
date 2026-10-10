import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
T = Path(r"D:\tmp_timing")
PKG = Path(r"D:\models\pkg_TinyRefV4us0"); (PKG / "model_dir").mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def R(name, cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    print("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)), flush=True)
    if r.returncode != 0: print("   ", (r.stderr or "")[-300:], flush=True)
    return r.returncode
R("export_us0", [PY, "-u", str(HERE/"opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                 "--refiner", r"D:\models\refine_v1\refiner_thin2_s6000.pt", "--modeldir", str(PKG/"model_dir"), "--unsharp", "0"])
R("render100", [PY, "-u", str(HERE/"opt11_render_pkg.py"), "--modeldir", str(PKG/"model_dir"), "--out", str(PKG/"output_dir")])
R("package", [PY, str(T/"package_submit_v2.py"), "--out-dir", str(PKG/"output_dir"), "--model-dir", str(PKG/"model_dir"),
              "--name", "TinyRefV4us0", "--zip", r"D:\models\opt11\Track2_TinyRefV4us0.zip"])
print("BUILD_V4US0_DONE", flush=True)
