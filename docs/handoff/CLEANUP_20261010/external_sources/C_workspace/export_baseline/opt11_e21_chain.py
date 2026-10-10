r"""E21 链：可微 CLIP-IQA 奖励的 refiner（两路：从头训 / 从 V4 refiner 微调），随后导出+渲染+代理+严格时延。
判据（预登记）：clipiqa@4K（主）、clipiqa@512crop（辅，检查尺度一致性）、niqe/musiq/lapvar（防奖励攻击）、
              同会话 lat3 median；折算 b∈{0.34, 0.97} 的总分比 vs V4。
"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
CK = Path(r"D:\models\refine_v1"); LOG = Path(r"D:\models\opt11\e21_chain.log")
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sh(name, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-2:]
    log("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)))
    if r.returncode != 0: log("   stderr: " + (r.stderr or "")[-400:])
    return r.returncode
TGT = r"D:\datasets\distill256\tgt"; STU = r"D:\models\tiny_v1\v21_s21000.pt"
T = str(HERE / "opt11_train_refiner_clip.py")
RUNS = [
  ("clip1", ["--steps","6000","--ch","16","--blocks","2","--w_clip","5","--w_gan","0.3","--tgt_dir",TGT]),
  ("clip2", ["--steps","3000","--ch","16","--blocks","2","--w_clip","20","--w_gan","0.3","--init",str(CK/"refiner_thin2_s6000.pt"),"--tgt_dir",TGT]),  # 剂量-响应：强权重从 V4 refiner 微调
]
log("== E21 链启动 ==")
for tag, extra in RUNS:
    sh("train_" + tag, [PY, "-u", T, "--tag", tag] + extra, 5400)
for tag, _ in RUNS:
    ck = None
    for st in (6000, 3000):
        p = CK / ("refiner_%s_s%d.pt" % (tag, st))
        if p.exists(): ck = p; break
    if ck is None: log("缺检查点 " + tag); continue
    sh("export_" + tag, [PY, "-u", str(HERE/"opt11_export_combined.py"), "--ckpt", STU, "--refiner", str(ck),
        "--modeldir", r"D:\models\pkg_%s\model_dir" % tag, "--unsharp", "60", "--in_res", "256"], 900)
dirs = [r"D:\models\pkg_TinyRefV4\model_dir"] + [r"D:\models\pkg_%s\model_dir" % t for t, _ in RUNS if (CK/("refiner_%s_s%d.pt" % (t, 6000 if t=="clip1" else 3000))).exists()]
sh("lat3", [PY, "-u", str(HERE/"opt11_lat3.py")] + dirs + ["--rounds","3","--n","20","--preheat","45"], 1800)
for tag, _ in RUNS:
    d = r"D:\models\opt11\%s_render" % tag
    sh("render_" + tag, [PY, "-u", str(HERE/"opt11_render_pkg.py"), "--modeldir", r"D:\models\pkg_%s\model_dir" % tag,
        "--out", d, "--n", "30", "--order", "lex"], 1800)
    sh("suite_" + tag, [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", d, "--tag", "E21_" + tag], 1800)
    sh("cropclip_" + tag, [PY, "-u", str(HERE/"opt11_crop_clip.py"), "--dir", d, "--tag", "E21_" + tag], 1800)
log("E21_CHAIN_DONE")
