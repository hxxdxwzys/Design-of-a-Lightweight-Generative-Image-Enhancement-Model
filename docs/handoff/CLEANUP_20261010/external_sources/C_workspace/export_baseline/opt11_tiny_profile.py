import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
md = r"D:\models\pkg_Tinyv21\model_dir"
spec = importlib.util.spec_from_file_location("rm", os.path.join(md, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(md)
dev = R.device

def timeit(fn, n=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return statistics.median(ts)

with torch.no_grad():
    full = timeit(lambda: R.infer(x, prompt=""))
    # GPU busy（CUDA event）for full call
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record(); R.infer(x, prompt=""); e.record(); torch.cuda.synchronize()
    busy = s.elapsed_time(e)
    # 各段：fp16 输入/下采样/模型(图重放)/上采样/clamp
    xh = x.to(dev, dtype=torch.float16)
    down = timeit(lambda: F.interpolate(xh, size=(256, 256), mode="bicubic", align_corners=False))
    xs = F.interpolate(xh, size=(256, 256), mode="bicubic", align_corners=False)
    R.static_in.copy_(xs)
    replay = timeit(lambda: R.graph.replay())
    y = R.static_out.clone()
    up = timeit(lambda: F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False))
    todev = timeit(lambda: x.to(dev, dtype=torch.float16))
    yup = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
    clamp = timeit(lambda: yup.float().clamp(-1, 1))
    # 模型 eager（无图）
    net = R.m
    eager = timeit(lambda: net(xs))
print("full infer      : %.2f ms   (GPU busy %.2f ms)" % (full, busy))
print("  x.to(cuda,fp16): %.2f ms" % todev)
print("  downsample     : %.2f ms" % down)
print("  graph replay   : %.2f ms" % replay)
print("  model eager    : %.2f ms" % eager)
print("  upsample       : %.2f ms" % up)
print("  clamp/float    : %.2f ms" % clamp)
print("  合计(分段)     : %.2f ms" % (todev + down + replay + up + clamp))
