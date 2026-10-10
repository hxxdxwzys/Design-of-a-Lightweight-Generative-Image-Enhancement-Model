# -*- coding: utf-8 -*-
"""B1' 评测链：导出(bcap refiner + 学生 + unsharp thr=1) -> 渲染 30 张 -> 评测 -> 判决。
用法: python b_cap_chain.py <bcap_ckpt> <tag>
判据（预登记，冻结）：
  全链 clipiqa > 0.8118  => 净胜最优（V4t1）
  > 0.8042               => 优于已提交 V4
  <= 0.8042              => 容量轴在充分训练下也 CLOSED
"""
import json, os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
CK = sys.argv[1]; TAG = sys.argv[2]
MD = Path(r"D:\models") / ("pkg_" + TAG)
EVAL = Path(r"D:\models\opt11") / ("eval_" + TAG)
METRICS = Path(r"D:\models\opt11\test30_metrics.json")
BEST, V4 = 0.8118, 0.8042


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


print("[chain] B1' 评测链 tag=%s ckpt=%s" % (TAG, CK), flush=True)
rc = run("export", [PY, "-u", str(T / "opt11_export_bcap.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                    "--refiner", CK, "--modeldir", str(MD), "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
if rc != 0: sys.exit(1)
rc = run("render30", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                      "--modeldir", str(MD), "--out", str(EVAL), "--n", "30"], timeout=7200)
if rc != 0: sys.exit(1)
run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(EVAL), "--n", "30"], timeout=3600)

m = json.loads(METRICS.read_text(encoding="utf-8"))
c, q = float(m["clipiqa"]), float(m["niqe"])
v = "WIN(>V4t1)" if c > BEST else ("BETTER_THAN_V4" if c > V4 else "CLOSED")
print("\n[chain] ===== 判决 =====", flush=True)
print("[chain] %s clipiqa=%.4f niqe=%.4f  => %s" % (TAG, c, q, v), flush=True)
print("[chain] 参照 V4t1=0.8118/4.8555 ; 已提交V4=0.8042/4.9967", flush=True)
(Path(r"D:\models\opt11") / ("verdict_%s.json" % TAG)).write_text(
    json.dumps({"tag": TAG, "clipiqa": c, "niqe": q, "verdict": v}, indent=1), encoding="utf-8")
print("BCAP_CHAIN_DONE", TAG, v, flush=True)
