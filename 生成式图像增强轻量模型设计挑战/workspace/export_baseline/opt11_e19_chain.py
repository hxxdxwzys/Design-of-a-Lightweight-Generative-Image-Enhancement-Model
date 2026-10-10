r"""E19 链：refiner 容量(4/8/16ch) × 质量(clipiqa) × 时延 的帕累托扫描。
同配方：--steps 6000 --bs 8 --lr 2e-4 --w_lpips 0.2 --w_gan 0.3 --blocks 2 --tgt_dir D:\datasets\distill256\tgt
导出：学生 v21 + refiner + unsharp60（in_res 256）；对照锚：V4(旧16ch thin2)、L256(无 refiner)
判据（预登记）：同会话 lat3 median + clipiqa@4K/niqe@4K/musiq@512 代理；用 b∈{0.34(新时代族内), 0.97(跨族)} 折算总分比。
"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
CK = Path(r"D:\models\refine_v1"); LOG = Path(r"D:\models\opt11\e19_chain.log")
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sh(name, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    log("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)))
    if r.returncode != 0: log("   stderr: " + (r.stderr or "")[-300:])
    return r.returncode
TGT = r"D:\datasets\distill256\tgt"
STU = r"D:\models\tiny_v1\v21_s21000.pt"
SIZES = [4, 8, 16]
log("== E19 链启动 ==")
for ch in SIZES:
    sh("train_r%02d" % ch, [PY, "-u", str(HERE/"opt11_train_refiner_gan.py"), "--tag", "r%02d" % ch,
        "--ch", str(ch), "--blocks", "2", "--steps", "6000", "--bs", "8", "--lr", "2e-4",
        "--w_lpips", "0.2", "--w_gan", "0.3", "--tgt_dir", TGT], 3600)
for ch in SIZES:
    ck = CK / ("refiner_r%02d_s6000.pt" % ch)
    dst = r"D:\models\pkg_R%02dch" % ch
    if not ck.exists(): log("缺检查点 " + str(ck)); continue
    sh("export_r%02d" % ch, [PY, "-u", str(HERE/"opt11_export_combined.py"), "--ckpt", STU, "--refiner", str(ck),
        "--modeldir", dst + r"\model_dir", "--unsharp", "60", "--in_res", "256"], 900)
# 严格同会话时延（含锚点）
dirs = [r"D:\models\pkg_TinyRefV4\model_dir", r"D:\models\pkg_L256\model_dir"] + [r"D:\models\pkg_R%02dch\model_dir" % ch for ch in SIZES]
sh("lat3", [PY, "-u", str(HERE/"opt11_lat3.py")] + dirs + ["--rounds", "3", "--n", "20", "--preheat", "45"], 1800)
for ch in SIZES:
    d = r"D:\models\opt11\R%02d_render" % ch
    sh("render_r%02d" % ch, [PY, "-u", str(HERE/"opt11_render_pkg.py"), "--modeldir", r"D:\models\pkg_R%02dch\model_dir" % ch,
        "--out", d, "--n", "30", "--order", "lex"], 1800)
    sh("suite_r%02d" % ch, [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", d, "--tag", "E19_R%02dch" % ch], 1800)
log("E19_CHAIN_DONE")
