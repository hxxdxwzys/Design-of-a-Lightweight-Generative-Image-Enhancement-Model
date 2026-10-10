r"""E14 链：同一 VOSR 观感目标下的容量/步数扫描。
A: 16ch 10k（省时延版）  B: 32ch 20k（延续 E8）  C: 48ch 20k（大容量）
每个跑完后渲染 30 张 + 全套代理，直接与 E8/V4 同协议比较。
"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
CK = Path(r"D:\models\refine_v1"); LOG = Path(r"D:\models\opt11\e14_chain.log")
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sh(name, cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    log("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)))
    if r.returncode != 0: log("   stderr: " + (r.stderr or "")[-300:])
    return r.returncode
TGT = str(CK / "tgt_vosr")
# e14b 用 --init 从 E8 的 10k 续训到 20k，与 E8 只差"步数"这一个变量（避免与容量混淆）
RUNS = [("e14a_16ch10k", ["--ch","16","--blocks","2","--steps","10000"]),
        ("e14b_32ch20k", ["--ch","32","--blocks","2","--steps","10000","--init", r"D:\models\refine_v1\refiner_vosr1_s10000.pt"]),
        ("e14c_48ch20k", ["--ch","48","--blocks","2","--steps","20000"])]
log("== E14 链启动 ==")
for tag, extra in RUNS:
    sh("train_" + tag, [PY, "-u", str(HERE/"opt11_train_refiner_gan.py"), "--tag", tag, "--lr", "2e-4",
                        "--w_gan", "0.12", "--tgt_dir", TGT] + extra, 7200)
    ck = CK / ("refiner_%s.pt" % tag)
    if ck.exists():
        sh("render_" + tag, [PY, "-u", str(HERE/"opt11_render_refined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                             "--refiner", str(ck), "--unsharp", "60", "--out", r"D:\models\opt11\%s_render" % tag, "--n", "30", "--order", "lex"], 1800)
        sh("suite_" + tag, [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", r"D:\models\opt11\%s_render" % tag, "--tag", "E14_" + tag], 1800)
    else:
        log("检查点缺失 " + str(ck))
log("E14_CHAIN_DONE")
