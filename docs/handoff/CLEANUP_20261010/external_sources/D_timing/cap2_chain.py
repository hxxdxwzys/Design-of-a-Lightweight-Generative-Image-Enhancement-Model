# -*- coding: utf-8 -*-
"""回归②：容量轴 —— 学生 32ch/0.87M -> 64ch/3.47M，其余配方逐项不动。
链：训练 30k 步（每 2000 存盘）-> 与历史逐项同口径评测（字典序 30 张、down=2、4K 全图代理）。
判据（预登记，跑完不改）：
  学生单独 clipiqa@4K >= 0.70  => 容量轴活，值得再接 refiner
                       < 0.65  => 容量轴死，转 unsharp 参数扫
  参照：v21 学生 = clipiqa 0.6306 / niqe 5.4843 / lapvar 164.2（同口径）
"""
import subprocess, sys, time
from pathlib import Path

PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
TAG = "cap2"
STEPS = 30000

env_note = "capacity axis: ch=64,64,64,128 (3.47M, 4.0x v21) bs=8 steps=30000, 其余与 v11b..v21 配方逐项相同"


def run(name, cmd, timeout=36000):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-6:]
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in tail:
        print("        " + l, flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1500:], flush=True)
    return r.returncode


print("[chain] %s" % env_note, flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

rc = run("train", [PY, "-u", str(T / "train_tiny_gan.py"),
                   "--ch", "64,64,64,128", "--res", "256", "--loss_res", "512", "--d_band", "1",
                   "--bs", "8", "--steps", str(STEPS), "--w_gan", "0.1", "--tag", TAG])
ckpt = Path(r"D:\models\tiny_v1") / ("%s_s%d.pt" % (TAG, STEPS))
if rc != 0 or not ckpt.exists():
    print("[chain] 训练未产出 %s -> 停止" % ckpt, flush=True)
    sys.exit(1)

rc = run("eval_student", [PY, "-u", str(EB / "opt11_eval_student.py"), TAG, str(ckpt)], timeout=7200)

mp = Path(r"D:\models\opt11") / ("test30_%s.json" % TAG)
if mp.exists():
    print("\n[chain] ===== RESULT =====", flush=True)
    print("[chain] %s : %s" % (TAG, mp.read_text(encoding="utf-8").replace("\n", " ")), flush=True)
    print("[chain] 参照 v21 : clipiqa 0.6306 / niqe 5.4843 / lapvar 164.2", flush=True)
print("[chain] done %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("CAPACITY_AXIS_CHAIN_DONE", flush=True)
