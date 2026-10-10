r"""交付包可运行性验证：新进程加载 model_dir，512 输入跑通并计时；打印依赖与哈希。"""
import hashlib, importlib.util, json, os, statistics, sys, time
from pathlib import Path
import torch
MD = r"D:\models\pkg_TinyRefV4\model_dir"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
x = torch.rand(1, 3, 512, 512) * 2 - 1
spec = importlib.util.spec_from_file_location("r", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
with torch.no_grad():
    for _ in range(2): y = R.infer(x, prompt="")
    ts = []
    for _ in range(10):
        t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
print("输出 shape", tuple(y.shape), "范围 [%.3f, %.3f]" % (float(y.min()), float(y.max())))
print("冷启动 512 时延 median %.2f ms (best %.2f)" % (statistics.median(ts), min(ts)))
mods = sorted({k.split(".")[0] for k in sys.modules if not k.startswith("_")})
heavy = [k for k in ("torch", "torchvision", "numpy", "PIL", "pyiqa", "diffusers", "lpips", "transformers") if k in mods]
print("本进程实际加载的相关第三方包:", heavy)
for f in sorted(Path(MD).glob("*")):
    if f.is_file():
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        print("  %-16s %9d B  sha256:%s" % (f.name, f.stat().st_size, h))
