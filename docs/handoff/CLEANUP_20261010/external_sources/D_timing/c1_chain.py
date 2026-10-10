# -*- coding: utf-8 -*-
"""回归⑥（C1）：容量 + refiner 全链。
单变量：学生容量 0.87M -> 13.8M（16x），其余配方逐项不动；随后用同一 thin 配方重训 refiner。

步骤：
  1) 训 256 域 16x 容量学生        (ch=128,128,128,256, res256/loss512/d_band1/bs8/30k)
  2) 生成该学生的 refiner 训练输入 (gen_refine_data2)
  3) 用 thin 配方训 refiner        (16ch/2bl, 8k@w0.1 + 6k@w0.3)
  4) 全链评测                      (opt11_render_pkg + measure_test30)
  5) 512 时延                      (lat_ckpt)

预登记判据（跑完不改）：全链 clipiqa >= 0.83 => 轴活；< 0.8042 => 相对已提交件无增益，轴死。
对照：已提交 V4 全链 clipiqa 0.8042 / niqe 4.9967；V4 的 refiner 增益 = +0.1736。
"""
import os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
STU = "cap3"
REF = "ref_cap3"
STEPS_S = 30000
CH = "128,128,128,256"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")

print("[chain] C1 capacity+refiner: student ch=%s (16x), then thin refiner on ITS output" % CH, flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def run(name, cmd, timeout=43200):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-3:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1200:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


# 1) 学生
rc = run("train_student_cap3", [PY, "-u", str(T / "train_tiny_gan.py"),
                                "--ch", CH, "--res", "256", "--loss_res", "512", "--d_band", "1",
                                "--bs", "8", "--steps", str(STEPS_S), "--w_gan", "0.1", "--tag", STU])
stu_ck = Path(r"D:\models\tiny_v1") / ("%s_s%d.pt" % (STU, STEPS_S))
if rc != 0 or not stu_ck.exists():
    print("[chain] 学生未产出 -> 停止", flush=True); sys.exit(1)

# 2) 生成 refiner 训练输入
in_dir = Path(r"D:\models\refine_v1") / ("in_" + STU)
rc = run("gen_refine_data", [PY, "-u", str(T / "gen_refine_data2.py"),
                             "--ckpt", str(stu_ck), "--out", str(in_dir)], timeout=7200)
if rc != 0 or not in_dir.exists() or len(list(in_dir.glob("*.jpg"))) < 100:
    print("[chain] refiner 输入不足 -> 停止", flush=True); sys.exit(1)

# 3) refiner：thin 配方（16ch/2bl，8k@w_gan0.1 + 6k@w_gan0.3）
rc = run("train_refiner_a", [PY, "-u", str(T / "train_refiner_gan2.py"),
                             "--in_dir", str(in_dir), "--tgt_dir", r"D:\datasets\distill256\tgt",
                             "--ch", "16", "--blocks", "2", "--steps", "8000",
                             "--w_gan", "0.1", "--tag", REF + "a"], timeout=10800)
rc = run("train_refiner_b", [PY, "-u", str(T / "train_refiner_gan2.py"),
                             "--in_dir", str(in_dir), "--tgt_dir", r"D:\datasets\distill256\tgt",
                             "--ch", "16", "--blocks", "2", "--steps", "6000",
                             "--w_gan", "0.3", "--lr_d", "1e-4", "--tag", REF + "b",
                             "--init", r"D:\models\refine_v1\refiner_%sa_s8000.pt" % REF], timeout=10800)
ref_ck = Path(r"D:\models\refine_v1") / ("refiner_%sb_s6000.pt" % REF)
if not ref_ck.exists():
    print("[chain] refiner 未产出 -> 停止", flush=True); sys.exit(1)

# 4) 全链评测（与已提交 V4 完全同口径：打包 runner + thr=1 最优档 + 字典序 30 张）
md = Path(r"D:\models\pkg_cap3ref")
rc = run("export", [PY, "-u", str(EB / "opt11_export_us.py"), "--ckpt", str(stu_ck),
                    "--refiner", str(ref_ck), "--modeldir", str(md),
                    "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
outd = Path(r"D:\models\opt11") / ("eval_" + REF)
rc = run("render30", [PY, "-u", str(EB / "opt11_render_pkg.py"), "--modeldir", str(md), "--out", str(outd), "--n", "30"], timeout=7200)
rc = run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(outd), "--n", "30"], timeout=3600)

mp = Path(r"D:\models\opt11\test30_metrics.json")
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists():
    print("[chain] full chain: %s" % mp.read_text(encoding="utf-8").replace("\n", " "), flush=True)
print("[chain] refs: SUBMITTED V4 = clipiqa 0.8042 / niqe 4.9967 ; V4 refiner gain = +0.1736", flush=True)
print("[chain] criteria: >=0.83 alive ; <0.8042 dead", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("C1_CHAIN_DONE", flush=True)
