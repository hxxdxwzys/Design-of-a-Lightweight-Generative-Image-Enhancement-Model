# -*- coding: utf-8 -*-
"""定位 CUDA graph 崩溃：是 S1b 模型不可捕获，还是 runner 的捕获方式问题？"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dev = "cuda"
MD_S1B = r"D:\models\pkg_s1b_smoke"
MD_V4 = r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")

img = np.asarray(Image.open(TEST / "case3.jpg").convert("RGB"), dtype=np.uint8)[:512, :512]
t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)


def load_runner(md):
    spec = importlib.util.spec_from_file_location("r_%d" % abs(hash(md)), os.path.join(md, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_graph(md, label):
    m = load_runner(md)
    R = m.Runner(md)
    print("[%s] graph=%s" % (label, R.graph is not None), flush=True)
    with torch.no_grad():
        y = R.infer(t.clone(), prompt="")
    print("   infer OK out=%s range=[%.3f,%.3f]" % (tuple(y.shape), float(y.min()), float(y.max())), flush=True)
    return R


def test_manual_capture(md, label):
    """手动尝试捕获，看具体哪一步失败。"""
    mm = torch.jit.load(os.path.join(md, "your_model.pt"), map_location=dev); mm.eval()
    si = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                mm(si)
        torch.cuda.current_stream().wait_stream(s)
        print("[%s] 手动 warmup OK" % label, flush=True)
    except Exception as e:
        print("[%s] 手动 warmup FAILED: %s" % (label, str(e)[:200]), flush=True); return
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            so = mm(si)
        print("[%s] 手动 capture OK" % label, flush=True)
    except Exception as e:
        print("[%s] 手动 capture FAILED: %s" % (label, str(e)[:300]), flush=True)


print("=== V4（现役，已知可建图）===", flush=True)
test_manual_capture(MD_V4, "V4")
print("\n=== S1b（新导出）===", flush=True)
test_manual_capture(MD_S1B, "S1b")
print("\n=== 两者 eager 输出对比 ===", flush=True)
m1 = torch.jit.load(os.path.join(MD_V4, "your_model.pt"), map_location=dev); m1.eval()
m2 = torch.jit.load(os.path.join(MD_S1B, "your_model.pt"), map_location=dev); m2.eval()
x = t.to(dev).half()
with torch.no_grad():
    a = m1(x).float().cpu().numpy(); b = m2(x).float().cpu().numpy()
print("V4 vs S1b: max=%.5f mean=%.6f" % (np.abs(a - b).max(), np.abs(a - b).mean()), flush=True)
print("(S1b 是 s2000 步、U 已开始偏离恒等；差异应很小但不为 0)", flush=True)
print("DIAG_DONE", flush=True)
