# -*- coding: utf-8 -*-
"""回归③：内置 unsharp 三参数扫描（零时延代价的后处理旋钮）。
项目历史上只试过强度 60% 与 100% 两点；σ(1.2) 与阈值(3/255) 从未扫过。
口径：与历史完全一致 —— 用打包后的 runner 渲染字典序 30 张 4K → measure_test30。
用法: python us_sweep.py
"""
import subprocess, sys, os, time, json
from pathlib import Path

PY = r"D:\Python\python.exe"
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
T = Path(r"D:\tmp_timing")
REFINER = r"D:\models\refine_v1\refiner_thin2_s6000.pt"
STUDENT = r"D:\models\tiny_v1\v21_s21000.pt"
ROOT = Path(r"D:\models\pkg_us")

# (tag, unsharp_pct, sigma, thr_u8) —— 基线 = 已提交 V4 的配方
CONFIGS = [
    ("us60_s12_t3", 60.0, 1.2, 3.0),    # 已提交 V4 的确切配方（基准）
    ("us40_s12_t3", 40.0, 1.2, 3.0),
    ("us80_s12_t3", 80.0, 1.2, 3.0),
    ("us60_s08_t3", 60.0, 0.8, 3.0),
    ("us60_s20_t3", 60.0, 2.0, 3.0),
    ("us60_s12_t1", 60.0, 1.2, 1.0),
    ("us60_s12_t6", 60.0, 1.2, 6.0),
]

print("[us_sweep] %d configs, student=%s refiner=%s" % (len(CONFIGS), Path(STUDENT).name, Path(REFINER).name), flush=True)
results = {}
for tag, pct, sigma, thr in CONFIGS:
    md = ROOT / tag
    print("\n[us_sweep] === %s  unsharp=%s sigma=%s thr=%s ===" % (tag, pct, sigma, thr), flush=True)
    r = subprocess.run([PY, "-u", str(EB / "opt11_export_us.py"),
                        "--ckpt", STUDENT, "--refiner", REFINER, "--modeldir", str(md),
                        "--unsharp", str(pct), "--sigma", str(sigma), "--thr", str(thr)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if r.returncode != 0:
        print("   export FAILED:", (r.stderr or "")[-400:], flush=True); continue
    out = Path(r"D:\models\opt11") / ("eval_" + tag)
    r = subprocess.run([PY, "-u", str(EB / "opt11_render_pkg.py"), "--modeldir", str(md), "--out", str(out), "--n", "30"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        print("   render FAILED:", (r.stderr or "")[-400:], flush=True); continue
    r = subprocess.run([PY, "-u", str(T / "measure_test30.py"), "--dir", str(out), "--n", "30"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    txt = [l for l in (r.stdout or "").splitlines() if l.strip().startswith("[measure]")]
    print("   " + (" | ".join(txt) if txt else "measure FAILED"), flush=True)
    if txt:
        try:
            results[tag] = json.loads(txt[-1].split("[measure]", 1)[1].strip())
        except Exception:
            pass

print("\n[us_sweep] ===== 汇总 =====", flush=True)
print("%-16s %10s %10s" % ("config", "clipiqa", "niqe"), flush=True)
for tag, _, _, _ in CONFIGS:
    if tag in results:
        print("%-16s %10.4f %10.4f" % (tag, results[tag]["clipiqa"], results[tag]["niqe"]), flush=True)
print("[us_sweep] 参照：已提交 V4 = clipiqa 0.8046 / niqe 5.0020", flush=True)
Path(r"D:\models\opt11\us_sweep.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("US_SWEEP_DONE", flush=True)
