r"""整链成本拆解：无 refiner / 16ch / 32ch / r256(32ch2bl) 四种导出，各测一次（同进程、独立 runner）。"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
x = SB.load_input("c1_512")
CKPT = r"D:\models\tiny_v1\v21_s21000.pt"
variants = [
    ("noref", ""),
    ("thin16", r"D:\models\refine_v1\refiner_thin2_s6000.pt"),
    ("w32", r"D:\models\refine_v1\refiner_w32_s10000.pt"),
]
def measure(tag, refiner):
    MD = str(Path(r"D:\models") / ("pkg_cost_" + tag) / "model_dir")
    cmd = [PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", CKPT, "--modeldir", MD, "--unsharp", "60"]
    if refiner: cmd += ["--refiner", refiner]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    if r.returncode != 0:
        print("[%s] export FAIL %s" % (tag, (r.stderr or "")[-200:]), flush=True); return
    spec = importlib.util.spec_from_file_location("c_" + tag, os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(MD)
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(20):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    print("[%-8s] median %.2f ms best %.2f" % (tag, statistics.median(ts), min(ts)), flush=True)
    del R; torch.cuda.empty_cache()
for tag, ref in variants:
    measure(tag, ref)
# 256 域 refiner 形状（随机权重，仅测成本）
import torch.nn as nn, torch.nn.functional as F
torch.save({"net": nn.Sequential(nn.Conv2d(3,32,3,1,1), nn.ReLU(True), nn.Conv2d(32,32,3,1,1), nn.ReLU(True), nn.Conv2d(32,12,3,1,1)).state_dict(),
            "ch": 32, "blocks": 2, "steps": 0, "arch": "r256"}, r"D:\models\refine_v1\refiner_r256probe.pt")
measure("r256_32", r"D:\models\refine_v1\refiner_r256probe.pt")
print("COST_DECOMP_DONE", flush=True)
