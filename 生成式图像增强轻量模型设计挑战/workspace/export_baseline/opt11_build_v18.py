import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
PKG = Path(r"D:\models\pkg_TinyV18")
(PKG / "model_dir").mkdir(parents=True, exist_ok=True)
(PKG / "output_dir").mkdir(parents=True, exist_ok=True)
CKPT = r"D:\models\tiny_v1\v18_s25000.pt"

def run(name, cmd, timeout=2400):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-3:]
    print("[%s] rc=%d %.1fs :: %s" % (name, r.returncode, time.time() - t0, " | ".join(tail)), flush=True)
    if r.returncode != 0:
        print("   stderr:", (r.stderr or "")[-600:], flush=True)
    return r.returncode

env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def R(name, cmd, **kw):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, **kw)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-3:]
    print("[%s] rc=%d %.1fs :: %s" % (name, r.returncode, time.time() - t0, " | ".join(tail)), flush=True)
    if r.returncode != 0:
        print("   stderr:", (r.stderr or "")[-500:], flush=True)
    return r.returncode

R("export_tiny", [PY, "-u", str(T / "export_tiny.py"), "--ckpt", CKPT, "--modeldir", str(PKG / "model_dir"), "--down", "2"], timeout=1200)
R("render_100", [PY, "-u", str(T / "render_tiny.py"), "--ckpt", CKPT, "--down", "2", "--out", str(PKG / "output_dir")], timeout=3600)
n = len(list((PKG / "output_dir").glob("*.jpg")))
print("[render] output_dir jpg =", n, flush=True)
R("package", [PY, str(T / "package_submit_v2.py"), "--out-dir", str(PKG / "output_dir"),
              "--model-dir", str(PKG / "model_dir"), "--name", "TinyV18", "--zip", r"D:\models\opt11\Track2_TinyV18.zip"], timeout=1800)
