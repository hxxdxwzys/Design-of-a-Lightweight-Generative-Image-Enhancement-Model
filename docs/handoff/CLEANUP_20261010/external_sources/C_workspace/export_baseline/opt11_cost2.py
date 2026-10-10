r"""补测：无 refiner / r256(32ch,2bl) 两条对照（修正权重结构）。"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch, torch.nn as nn
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
x = SB.load_input("c1_512")
CKPT = r"D:\models\tiny_v1\v21_s21000.pt"

# 用导出脚本里的 Refiner 类构造 r256 权重（结构一致）
spec0 = importlib.util.spec_from_file_location("expc", str(HERE / "opt11_export_combined.py"))

def make_r256(ch=32, blocks=2):
    import importlib.util as iu
    # 直接从源码里取类：用一个最小 stub 避免执行 main
    src = (HERE / "opt11_export_combined.py").read_text(encoding="utf-8")
    cls_src = src.split("class Refiner(nn.Module):")[1].split("from diffusers import")[0]
    ns = {"nn": nn, "F": torch.nn.functional, "torch": torch}
    exec("class Refiner(nn.Module):" + cls_src, ns)
    m = ns["Refiner"](ch, blocks, "r256")
    torch.save({"net": m.state_dict(), "ch": ch, "blocks": blocks, "steps": 0, "arch": "r256"},
               r"D:\models\refine_v1\refiner_r256probe.pt")
    return r"D:\models\refine_v1\refiner_r256probe.pt"

def measure(tag, refiner):
    MD = str(Path(r"D:\models") / ("pkg_cost2_" + tag) / "model_dir")
    cmd = [PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", CKPT, "--modeldir", MD, "--unsharp", "60"]
    if refiner: cmd += ["--refiner", refiner]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    if r.returncode != 0:
        print("[%s] export FAIL: %s" % (tag, (r.stderr or "").strip().splitlines()[-1][:160]), flush=True); return
    spec = importlib.util.spec_from_file_location("c2_" + tag, os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(MD)
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(25):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    print("[%-10s] median %.2f ms best %.2f" % (tag, statistics.median(ts), min(ts)), flush=True)
    del R; torch.cuda.empty_cache()

measure("noref", "")
measure("r256_32x2", make_r256(32, 2))
print("COST2_DONE", flush=True)
