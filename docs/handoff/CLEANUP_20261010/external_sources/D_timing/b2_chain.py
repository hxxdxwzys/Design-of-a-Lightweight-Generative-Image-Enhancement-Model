# -*- coding: utf-8 -*-
"""B2（回归⑨）：可学习抗锯齿下采样，替换固定双三次 512->256。
这是链路上【唯一无参数、无学习】的环节，也是信息丢失点。

单变量：只换下采样方式（78 参数的可学习 5x5 高斯核）；ch/res/loss_res/d_band/bs/steps/w_gan 与 v21 逐项相同。

链：训练 30k -> 学生单独评测（学生档判据）-> 512 时延（确认代价）
预登记判据（跑完不改）：
  学生单独 clipiqa@4K >= 0.66  => 值得接 refiner 走全程（v21 学生 = 0.6306；cap2 4x容量 = 0.6656）
                      < 0.63  => 死亡
  时延：须 <= 8.5 ms（v21 学生同会话约 4.6-10.8 ms 漂移，故只做同会话相对比较）
"""
import os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
TAG = "lrnd"
STEPS = 30000

print("[chain] B2: learnable anti-aliased downsampler replaces fixed bicubic 512->256", flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def run(name, cmd, timeout=43200):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-4:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1200:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


rc = run("train_lrnd", [PY, "-u", str(T / "train_tiny_lrndown.py"), "--lrn_down", "1",
                        "--ch", "32,32,32,64", "--res", "256", "--loss_res", "512", "--d_band", "1",
                        "--bs", "6", "--steps", str(STEPS), "--w_gan", "0.1", "--tag", TAG])
ck = Path(r"D:\models\tiny_v1") / ("%s_s%d.pt" % (TAG, STEPS))
if rc != 0 or not ck.exists():
    print("[chain] 学生未产出 %s -> 停止" % ck, flush=True); sys.exit(1)

run("eval_student", [PY, "-u", str(T / "b2_eval.py"), str(ck), TAG, "--n", "30"], timeout=7200)
run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(Path(r"D:\models\opt11") / ("eval_" + TAG)), "--n", "30"], timeout=3600)

mp = Path(r"D:\models\opt11\test30_metrics.json")
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists():
    print("[chain] lrnd student alone: %s" % mp.read_text(encoding="utf-8").replace("\n", " "), flush=True)
print("[chain] refs: v21 student = 0.6306/5.4843 ; cap2(4x cap) student = 0.6656/5.853", flush=True)
print("[chain] criteria: >=0.66 alive (go to refiner) ; <0.63 dead", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("B2_CHAIN_DONE", flush=True)
