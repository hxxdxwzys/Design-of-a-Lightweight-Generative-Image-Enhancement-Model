# -*- coding: utf-8 -*-
"""回归⑤（C）：512 原生域学生 —— 解除「在 256 域前向、却在 512 域算损失」的域变配。

单变量：--res 512 --loss_res 512（原配方 --res 256 --loss_res 512），其余逐项不动。
容量保持同一家族形状 ch=64,64,64,128（3.47M，与 cap2 同参数量）以便与 cap2/ v21 三方对照。

预登记判据（跑完不改）：
  学生单独 clipiqa@4K >= 0.70  => 域轴活，值得接 refiner
                     < 0.66  => 域轴死
  对照：v21(256域) = 0.6306 / cap2(256域,4x容量) = 0.6656 / E12(256训练但512推理) = 0.7018
"""
import os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
TAG = "d512"
STEPS = 20000
BS = 2

print("[chain] 512-native student: --res 512 --loss_res 512 --ch 64,64,64,128 --bs %d --steps %d" % (BS, STEPS), flush=True)
print("[chain] NOTE: bs 由 4 降到 2 —— 实测 bs=4 时 VRAM 7865/8188 MiB 触发换页颠簸"
      "(GPU 100%% 但仅 31 W, <100 步/27min)。bs=2 实测 6494 MiB / 75 W / 0.86 s/步。", flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def run(name, cmd, timeout=43200):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-4:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1200:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


rc = run("train_d512", [PY, "-u", str(T / "train_tiny_gan.py"),
                        "--res", "512", "--loss_res", "512", "--d_band", "1",
                        "--ch", "64,64,64,128", "--bs", str(BS), "--steps", str(STEPS),
                        "--w_gan", "0.1", "--tag", TAG])
ckpt = Path(r"D:\models\tiny_v1") / ("%s_s%d.pt" % (TAG, STEPS))
if rc != 0 or not ckpt.exists():
    print("[chain] 未产出 %s -> 停止" % ckpt, flush=True); sys.exit(1)

run("eval_student", [PY, "-u", str(EB / "opt11_eval_student.py"), TAG, str(ckpt)], timeout=7200)

mp = Path(r"D:\models\opt11") / ("test30_%s.json" % TAG)
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists():
    print("[chain] %s : %s" % (TAG, mp.read_text(encoding="utf-8").replace("\n", " ")), flush=True)
print("[chain] refs: v21(256)=0.6306/5.4843  cap2(256,4x)=0.6656/5.853  E12(512 infer)=0.7018", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("D512_CHAIN_DONE", flush=True)
