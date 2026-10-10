r"""E12：学生改在 512 域推理（in_res=512）+ 同一 refiner（V4/thin2），检验"瓶颈在学生"假设。
导出 -> 用打包 runner 渲染 test30 -> 代理套件 -> 时延（时延单独在空闲态测）。
用法: python opt11_e12_student512.py
"""
import importlib.util, os, subprocess, sys, time
from pathlib import Path
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
MD = r"D:\models\pkg_e12_stu512\model_dir"
def sh(name, cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-1:]
    print("[%s] rc=%d %.0fs :: %s" % (name, r.returncode, time.time()-t0, " | ".join(tail)), flush=True)
    if r.returncode != 0: print("   stderr:", (r.stderr or "")[-300:], flush=True)
    return r.returncode
sh("export_in512", [PY, "-u", str(HERE/"opt11_export_combined.py"), "--ckpt", r"D:\models\tiny_v1\v21_s21000.pt",
                    "--refiner", r"D:\models\refine_v1\refiner_thin2_s6000.pt", "--modeldir", MD,
                    "--unsharp", "60", "--in_res", "512"], 1800)
sh("render30", [PY, "-u", str(HERE/"opt11_render_pkg.py"), "--modeldir", MD, "--out", r"D:\models\opt11\e12_render", "--n", "30", "--order", "lex"], 1800)
print("E12_RENDER_DONE", flush=True)
