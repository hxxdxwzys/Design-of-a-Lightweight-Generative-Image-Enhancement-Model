r"""E17 链：时延-质量曲线（在我们自己的部署区间内找拐点）。
变体：A=学生256域+unsharp60（无refiner） B=192域+unsharp C=128域+unsharp；对照 D=V4(带refiner) E=V21(学生裸,无unsharp)
判据（预登记）：严格同会话时延(lat3) + clipiqa@4K/niqe@4K/musiq@512 代理；
      用旧时代 clipiqa 律 IT=43.4156·c^0.9739 折算预测分（该律对新时代 5 点误差 ±4.5%）。
"""
import os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
LOG = Path(r"D:\models\opt11\e17_chain.log")
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
CK = r"D:\models\tiny_v1\v21_s21000.pt"
VAR = [("L256", 256, r"D:\models\pkg_L256"), ("L192", 192, r"D:\models\pkg_L192"), ("L128", 128, r"D:\models\pkg_L128")]
log("== E17 链启动 ==")
for tag, res, dirp in VAR:
    sh("export_" + tag, [PY, "-u", str(HERE/"opt11_export_combined.py"), "--ckpt", CK, "--modeldir", dirp + r"\model_dir",
                         "--unsharp", "60", "--in_res", str(res)], 900)
# 严格同会话时延：D(V4) E(V21) A B C 交替
dirs = [r"D:\models\pkg_TinyRefV4\model_dir", r"D:\models\pkg_TinyV21\model_dir"] + [d + r"\model_dir" for _, _, d in VAR]
sh("lat3", [PY, "-u", str(HERE/"opt11_lat3.py")] + dirs + ["--rounds", "3", "--n", "20", "--preheat", "45"], 1800)
# 渲染 + 代理
for tag, res, dirp in VAR:
    sh("render_" + tag, [PY, "-u", str(HERE/"opt11_render_pkg.py"), "--modeldir", dirp + r"\model_dir",
                         "--out", r"D:\models\opt11\%s_render" % tag, "--n", "30", "--order", "lex"], 1800)
    sh("suite_" + tag, [PY, "-u", str(HERE/"opt11_proxy_suite.py"), "--dir", r"D:\models\opt11\%s_render" % tag,
                        "--tag", "E17_" + tag], 1800)
log("E17_CHAIN_DONE")
