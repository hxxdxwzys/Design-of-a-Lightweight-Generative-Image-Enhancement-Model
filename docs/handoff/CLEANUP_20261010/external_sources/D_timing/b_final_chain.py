# -*- coding: utf-8 -*-
"""B 路线终极链：等 bcap 训练完 -> ①bcap 评测 ②V3 同链重测 ③V4t1 同链重测 -> 三方同口径对比。
全程 GPU 串行；每步产物落盘，可中断续跑。
"""
import json, os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
OPT11 = Path(r"D:\models\opt11")
CK_CAP64 = Path(r"D:\models\refine_v1\bcap_cap64_s20000.pt")


def run(name, cmd, timeout=10800):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-2:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-600:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


def measure_and_log(tag):
    m = json.loads((OPT11 / "test30_metrics.json").read_text(encoding="utf-8"))
    c, q = float(m["clipiqa"]), float(m["niqe"])
    (OPT11 / ("test30_%s.json" % tag)).write_text(json.dumps({"clipiqa": c, "niqe": q, "n": 30}), encoding="utf-8")
    print("[chain] %-12s clipiqa=%.4f  niqe=%.4f" % (tag, c, q), flush=True)
    return c, q


# ---- 等待 bcap 训练完成 ----
deadline = time.time() + 4200
while time.time() < deadline and not CK_CAP64.exists():
    time.sleep(20)
if not CK_CAP64.exists():
    print("[chain] 等待 bcap 检查点超时 -> 退出", flush=True); sys.exit(1)
print("[chain] bcap 检查点就绪", flush=True)

res = {}

# ---- ① bcap: 64ch/4bl refiner + 学生 + thr=1 ----
MD = Path(r"D:\models\pkg_bcap64")
rc = run("export_bcap", [PY, "-u", str(T / "opt11_export_bcap.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                         "--refiner", str(CK_CAP64), "--modeldir", str(MD),
                         "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
if rc == 0:
    rc = run("render_bcap", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                             "--modeldir", str(MD), "--out", str(OPT11 / "eval_bcap64"), "--n", "30"], timeout=7200)
    if rc == 0:
        run("measure_bcap", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(OPT11 / "eval_bcap64"), "--n", "30"], timeout=3600)
        res["bcap64"] = measure_and_log("bcap64")

# ---- ② V3 同链重测（用其原始包，thr 保持 V3 原始配方）----
MD_V3 = Path(r"D:\models\pkg_TinyRefV3\model_dir")
if MD_V3.exists():
    run("clean_v3", [PY, "-c", "import shutil,os;shutil.rmtree(r'%s',ignore_errors=True)" % (MD_V3 / "__pycache__")], timeout=300)
    rc = run("render_v3", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                           "--modeldir", str(MD_V3), "--out", str(OPT11 / "eval_v3_samechain"), "--n", "30"], timeout=7200)
    if rc == 0:
        run("measure_v3", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(OPT11 / "eval_v3_samechain"), "--n", "30"], timeout=3600)
        res["V3"] = measure_and_log("V3")

# ---- ③ V4t1 同链重测（graph runner，与已提交 V4 同链）----
MD_V4T1 = Path(r"D:\models\pkg_v4t1_graph")
src_zip = OPT11 / "Track2_TinyRefV4t1.zip"
if src_zip.exists():
    import zipfile
    MD_V4T1.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip) as z:
        for n in z.namelist():
            if n.startswith("model_dir/") and not n.endswith("/"):
                (MD_V4T1 / Path(n).name).write_bytes(z.read(n))
    rc = run("render_v4t1", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                             "--modeldir", str(MD_V4T1), "--out", str(OPT11 / "eval_v4t1_samechain"), "--n", "30"], timeout=7200)
    if rc == 0:
        run("measure_v4t1", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(OPT11 / "eval_v4t1_samechain"), "--n", "30"], timeout=3600)
        res["V4t1"] = measure_and_log("V4t1")

print("\n[chain] ================= 三方同口径对比 =================", flush=True)
print("[chain] %-10s %10s %10s" % ("配置", "clipiqa", "niqe"), flush=True)
for k, (c, q) in res.items():
    print("[chain] %-10s %10.4f %10.4f" % (k, c, q), flush=True)
print("[chain] 已提交 V4（历史同链读数）= 0.8042 / 4.9967", flush=True)
(OPT11 / "bcap_final_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("B_CHAIN_DONE", flush=True)
