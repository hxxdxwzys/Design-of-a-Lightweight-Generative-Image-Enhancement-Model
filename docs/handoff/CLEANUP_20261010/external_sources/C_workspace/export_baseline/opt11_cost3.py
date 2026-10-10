r"""整链慢点拆解：无refiner@256 / 无refiner@192 / r256@256 / r256@256+benchmark。
（GPU 正被训练占用，故只做同场交替的相对对照，绝对值仅供参考。）"""
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
R256 = r"D:\models\refine_v1\refiner_r256probe.pt"     # 32ch/2bl, arch=r256（随机权重，仅测成本）

def export(tag, refiner, in_res, unsharp=60):
    MD = str(Path(r"D:\models") / ("pkg_cost3_" + tag) / "model_dir")
    cmd = [PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", CKPT, "--modeldir", MD,
           "--unsharp", str(unsharp), "--in_res", str(in_res)]
    if refiner: cmd += ["--refiner", refiner]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    if r.returncode != 0:
        print("[%s] export FAIL %s" % (tag, (r.stderr or "").strip().splitlines()[-1][:150]), flush=True); return None
    return MD

def load(MD, benchmark=False):
    if benchmark: torch.backends.cudnn.benchmark = True
    else: torch.backends.cudnn.benchmark = False
    spec = importlib.util.spec_from_file_location("cm_" + os.path.basename(os.path.dirname(MD)), os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Runner(MD)

def timeit(R, n=15):
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(n):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    return statistics.median(ts), min(ts)

variants = []
md = export("noref256", "", 256);        variants.append(("noref@256", md, False)); md and None
md = export("noref192", "", 192);        variants.append(("noref@192", md, False))
md = export("r256_256", R256, 256);      variants.append(("r256@256", md, False))
variants.append(("r256@256+bench", variants[-1][1], True))
for name, md, bench in variants:
    if not md: print("[%-16s] 跳过（导出失败）" % name, flush=True); continue
    try:
        R = load(md, bench); med, best = timeit(R)
        print("[%-16s] median %.2f ms best %.2f" % (name, med, best), flush=True)
        del R; torch.cuda.empty_cache()
    except Exception as e:
        print("[%-16s] 失败: %s" % (name, str(e)[:120]), flush=True)
print("COST3_DONE", flush=True)
