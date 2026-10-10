r"""E8 链：补完 VOSR 目标 -> 用 VOSR 观感训 refiner -> 评测（render + 全套代理）。"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
CK = Path(r"D:\models\refine_v1")
LOG = Path(r"D:\models\opt11\e8_chain.log")
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sh(name, cmd, timeout=5400):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    log("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)))
    if r.returncode != 0: log("   stderr: " + (r.stderr or "")[-300:])
    return r.returncode

log("== E8 链启动 ==")
sh("gen_targets", [PY, "-u", str(HERE/"opt11_gen_targets_vosr.py")], 5400)
n = len(list((CK/"tgt_vosr").glob("*.jpg"))); log("VOSR 目标 = %d" % n)
if n >= 2700:
    sh("train_vosr1", [PY, "-u", str(HERE/"opt11_train_refiner_gan.py"), "--steps", "10000", "--tag", "vosr1",
                       "--ch", "32", "--blocks", "2", "--lr", "2e-4", "--w_gan", "0.12",
                       "--tgt_dir", str(CK/"tgt_vosr")], 5400)
    E8 = CK/"refiner_vosr1_s10000.pt"
    if E8.exists():
        sh("render", [PY, "-u", str(HERE/"opt11_render_refined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                      "--refiner", str(E8), "--unsharp", "60", "--out", r"D:\models\opt11\e8_render", "--n", "30", "--order", "lex"], 1800)
        sh("suite", [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", r"D:\models\opt11\e8_render", "--tag", "E8_vosr1"], 1800)
    else:
        log("E8 检查点缺失")
else:
    log("目标不足（%d），跳过训练" % n)
log("E8_CHAIN_DONE")
