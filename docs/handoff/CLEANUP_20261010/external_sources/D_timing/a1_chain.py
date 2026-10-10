# -*- coding: utf-8 -*-
"""A1（回归⑧）：refiner 的【损失配方】轴 —— 唯一从未测过的 refiner 维度。
单变量：判别器 单尺度(512) -> 多尺度(512+256+128)，损失除以尺度数（量级不变）；
        输入/目标/参数/步数/w_gan 全部与已提交 V4 的 refiner 配方逐项相同。

预登记判据（跑完不改）：
  全链 clipiqa >= 0.83      => 配方轴活，值得继续深挖
  0.8160 ~ 0.83             => 优于现有最优档 V4t1(0.8118)，算正向
  <= 0.8042                 => 不如已提交件，配方轴死
参照（同口径 test30 / 同一条链）：
  已提交 V4        = clipiqa 0.8042 / niqe 4.9967
  V4t1(thr=1)      = clipiqa 0.8118 / niqe 4.8555   <- 当前最优
"""
import os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
IN = r"D:\models\refine_v1\in"          # 与 V4 的 refiner 完全同一批输入
TGT = r"D:\datasets\distill256\tgt"
REF = "ref_msd"

print("[chain] A1: multi-scale discriminator, everything else identical to V4's refiner recipe", flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def run(name, cmd, timeout=10800):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-4:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1200:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


rc = run("msd_a", [PY, "-u", str(T / "train_refiner_msd.py"), "--msd", "1",
                   "--in_dir", IN, "--tgt_dir", TGT, "--ch", "16", "--blocks", "2",
                   "--steps", "8000", "--w_gan", "0.1", "--tag", REF + "a"])
if rc != 0:
    print("[chain] phase A 失败 -> 停止", flush=True); sys.exit(1)
rc = run("msd_b", [PY, "-u", str(T / "train_refiner_msd.py"), "--msd", "1",
                   "--in_dir", IN, "--tgt_dir", TGT, "--ch", "16", "--blocks", "2",
                   "--steps", "6000", "--w_gan", "0.3", "--lr_d", "1e-4", "--tag", REF + "b",
                   "--init", r"D:\models\refine_v1\refiner_%sa_s8000.pt" % REF])
ref_ck = Path(r"D:\models\refine_v1") / ("refiner_%sb_s6000.pt" % REF)
if not ref_ck.exists():
    print("[chain] refiner 未产出 -> 停止", flush=True); sys.exit(1)

md = Path(r"D:\models\pkg_msdref")
run("export", [PY, "-u", str(EB / "opt11_export_us.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
               "--refiner", str(ref_ck), "--modeldir", str(md),
               "--unsharp", "60", "--sigma", "1.2", "--thr", "1.0"], timeout=3600)
outd = Path(r"D:\models\opt11") / ("eval_" + REF)
run("render30", [PY, "-u", str(EB / "opt11_render_pkg.py"), "--modeldir", str(md), "--out", str(outd), "--n", "30"], timeout=7200)
run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(outd), "--n", "30"], timeout=3600)

mp = Path(r"D:\models\opt11\test30_metrics.json")
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists():
    print("[chain] msd refiner full chain: %s" % mp.read_text(encoding="utf-8").replace("\n", " "), flush=True)
print("[chain] refs: V4 = 0.8042/4.9967 ; V4t1(best) = 0.8118/4.8555", flush=True)
print("[chain] criteria: >=0.83 alive ; >0.8118 positive ; <=0.8042 dead", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("A1_CHAIN_DONE", flush=True)
