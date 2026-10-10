# -*- coding: utf-8 -*-
"""回归⑥（C1'）：容量 + refiner 是否【可加】。
用【已有的 4x 容量学生 cap2】（训练已完成，无需重训），为其重训 refiner，测全链。

为什么这是决定性的：
  轴 2 只证伪了「更大容量【单独】有效」。本实验测「更大容量 + 为其重训的 refiner」全链。
  若 refiner 增益可加：cap2(0.6656) + 0.1736 = 0.8392 => 高于已提交 V4 的 0.8042
  若不可加：全链 <= 0.8042 => 容量轴在【全链】口径上也死，可彻底关闭。

预登记判据（跑完不改）：
  全链 clipiqa >= 0.83  => 可加，容量方向复活
  0.8042 ~ 0.83         => 部分可加，但需扣掉 cap2 的 2.32x 时延代价（factor -8%）后才判净值
  < 0.8042              => 容量轴 CLOSED（全链口径）

顺带对照：同一个 refiner 配方在 0.87M 学生上的增益 = +0.1736（0.6306 -> 0.8042）
"""
import os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
STU_CK = r"D:\models\tiny_v1\cap2_s30000.pt"
REF = "ref_cap2"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")

print("[chain] C1' capacity+refiner (reuse trained cap2 4x student)", flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def run(name, cmd, timeout=21600):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-3:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1000:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


in_dir = Path(r"D:\models\refine_v1") / "in_cap2"
rc = run("gen_refine_data(cap2)", [PY, "-u", str(T / "gen_refine_data2.py"),
                                   "--ckpt", STU_CK, "--out", str(in_dir)], timeout=7200)
n = len(list(in_dir.glob("*.jpg"))) if in_dir.exists() else 0
print("[chain] generated %d refiner inputs" % n, flush=True)
if n < 100:
    print("[chain] 输入不足 -> 停止", flush=True); sys.exit(1)

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

md = Path(r"D:\models\pkg_cap2ref")
rc = run("export", [PY, "-u", str(EB / "opt11_export_us.py"), "--ckpt", STU_CK,
                    "--refiner", str(ref_ck), "--modeldir", str(md),
                    "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
outd = Path(r"D:\models\opt11") / ("eval_" + REF)
rc = run("render30", [PY, "-u", str(EB / "opt11_render_pkg.py"), "--modeldir", str(md), "--out", str(outd), "--n", "30"], timeout=7200)
rc = run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(outd), "--n", "30"], timeout=3600)

mp = Path(r"D:\models\opt11\test30_metrics.json")
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists():
    print("[chain] cap2+refiner full chain: %s" % mp.read_text(encoding="utf-8").replace("\n", " "), flush=True)
print("[chain] refs: SUBMITTED V4 full chain = clipiqa 0.8042 / niqe 4.9967", flush=True)
print("[chain] cap2 student alone = 0.6656 ; refiner gain on 0.87M student = +0.1736", flush=True)
print("[chain] criteria: >=0.83 additive ; <0.8042 capacity axis CLOSED", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("C1P_CHAIN_DONE", flush=True)
