r"""夜间链：等 E6(r256d) 训完 -> 评测 E6 -> E8: 生成 VOSR 目标 -> 用 VOSR 目标训 refiner -> 评测 E8。
全部日志写 D:\models\opt11\night_chain.log；每步 rc 与被测读数都打印。
"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
LOG = Path(r"D:\models\opt11\night_chain.log")
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sh(name, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-1:]
    log("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)))
    if r.returncode != 0:
        log("   stderr: " + (r.stderr or "")[-300:])
    return r.returncode

CK = Path(r"D:\models\refine_v1")
E6 = CK / "refiner_r256d_s12000.pt"
log("== 夜间链启动：等待 E6 检查点 ==")
while not E6.exists():
    time.sleep(60)
log("E6 检查点就绪: %s" % E6.name)
time.sleep(20)

# --- 评测 E6 ---
sh("E6_render", [PY, "-u", str(HERE/"opt11_render_refined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                 "--refiner", str(E6), "--unsharp", "60", "--out", r"D:\models\opt11\e6_render", "--n", "30", "--order", "lex"], 1800)
sh("E6_suite", [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", r"D:\models\opt11\e6_render", "--tag", "E6_r256d_48ch"], 1800)

# --- E8: VOSR 目标 ---
sh("E8_gen_targets", [PY, "-u", str(HERE/"opt11_gen_targets_vosr.py")], 3600)
n = len(list((CK/"tgt_vosr").glob("*.jpg")))
log("E8 目标数量 = %d" % n)

# --- E8: 用 VOSR 目标训 refiner（512 域 32ch/2bl，与 V4 同形便于对比） ---
sh("E8_train", [PY, "-u", str(HERE/"opt11_train_refiner_gan.py"), "--steps", "10000", "--tag", "vosr1",
                "--ch", "32", "--blocks", "2", "--lr", "2e-4", "--w_gan", "0.12",
                "--tgt_dir", str(CK/"tgt_vosr")], 5400)

# --- E8: 评测 ---
E8 = CK / "refiner_vosr1_s10000.pt"
if E8.exists():
    sh("E8_render", [PY, "-u", str(HERE/"opt11_render_refined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                     "--refiner", str(E8), "--unsharp", "60", "--out", r"D:\models\opt11\e8_render", "--n", "30", "--order", "lex"], 1800)
    sh("E8_suite", [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", r"D:\models\opt11\e8_render", "--tag", "E8_vosr1"], 1800)
else:
    log("E8 检查点缺失，跳过评测")
log("NIGHT_CHAIN_DONE")
