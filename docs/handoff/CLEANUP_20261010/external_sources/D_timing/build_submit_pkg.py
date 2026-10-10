# -*- coding: utf-8 -*-
"""构建决赛预选提交包（官方格式）：output_dir/100 张 4K + model_dir/{your_model.pt, runner.py}。
用 V4t1 的配方从权重干净重建（thr=1），并落盘 manifest 以供人工复核逐字节核对。
"""
import json, os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
T = Path(r"D:\tmp_timing")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
PKG = Path(r"D:\models\pkg_submit_final")
STU = r"D:\models\tiny_v1\v21_s21000.pt"
REF = r"D:\models\refine_v1\refiner_thin2_s6000.pt"


def run(name, cmd, timeout=10800):
    print("[build] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=ENV)
    print("[build] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-2:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[build] STDERR:", (r.stderr or "")[-600:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


print("[build] 目标目录 %s" % PKG, flush=True)
if PKG.exists():
    import shutil
    shutil.rmtree(PKG, ignore_errors=True)
(PKG / "model_dir").mkdir(parents=True, exist_ok=True)
(PKG / "output_dir").mkdir(parents=True, exist_ok=True)

# 1) 导出模型（thr=1，与 V4t1 配方逐项一致）
rc = run("export", [PY, "-u", str(EB / "opt11_export_us.py"), "--ckpt", STU, "--refiner", REF,
                    "--modeldir", str(PKG / "model_dir"),
                    "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
if rc != 0: sys.exit(1)

# 2) 渲染 100 张 4K
rc = run("render100", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                       "--modeldir", str(PKG / "model_dir"), "--out", str(PKG / "output_dir")], timeout=10800)
if rc != 0: sys.exit(1)

# 3) manifest（供人工复核逐字节核对）
import hashlib
def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
       "recipe": {"student": os.path.basename(STU), "refiner": os.path.basename(REF),
                  "unsharp_pct": 60, "sigma": 1.2, "thr": 1.0},
       "model_dir": {}, "output_dir": {}}
for f in sorted((PKG / "model_dir").iterdir()):
    if f.is_file():
        man["model_dir"][f.name] = {"bytes": f.stat().st_size, "sha256": h(f)}
for f in sorted((PKG / "output_dir").glob("*.jpg")):
    man["output_dir"][f.name] = {"bytes": f.stat().st_size, "sha256": h(f)}
(PKG / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print("[build] manifest: %d jpg, model_dir=%s" % (len(man["output_dir"]), list(man["model_dir"])), flush=True)
print("BUILD_SUBMIT_DIR_DONE", flush=True)
