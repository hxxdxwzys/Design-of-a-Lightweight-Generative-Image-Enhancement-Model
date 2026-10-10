# -*- coding: utf-8 -*-
"""隔离：fp16 路径 + graph/非 graph 对照 + 跨进程确定性 + 与已提交出图对比。"""
import os, sys, importlib.util, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
S = os.path.join(P, "samples")
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
dev = "cuda"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run0"

img = np.asarray(Image.open(os.path.join(TEST, "case3.jpg")).convert("RGB"), dtype=np.uint8)
exp = np.asarray(Image.open(os.path.join(S, "case3_out.jpg")).convert("RGB"), dtype=np.uint8)


def prep_u8(p):
    """与 render/infer 完全一致：from_numpy -> permute -> [None], 再交给 runner 转 fp16。"""
    return torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)


def to_u8(y):
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


def cmp(n1, x, n2, y):
    dd = np.abs(x.astype(np.int16) - y.astype(np.int16))
    print("  %-28s vs %-28s max=%3d mean=%.4f pct!=0=%.2f%%" % (
        n1, n2, dd.max(), dd.mean(), (dd != 0).mean() * 100), flush=True)


spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

# --- full 4K render with the packaged runner (same as infer.render_4k) ---
sys.path.insert(0, P)
import infer as I

R = mm.Runner(MD)
print("[%s] graph available = %s" % (TAG, R.graph is not None), flush=True)
full = I.render_4k(R, img, I.make_feather(I.TILE, I.OVER), str(R.device))
h = hashlib.sha256(full.tobytes()).hexdigest()[:16]
print("[%s] 4K render sha256(uint8 bytes)=%s  mean=%.4f" % (TAG, h, full.mean()), flush=True)
cmp("fresh 4K render", full, "submitted case3_out.jpg", exp)
np.save(r"D:\tmp_render_%s.npy" % TAG, full)

# --- first tile only: graph vs no-graph ---
patch = img[0:512, 0:512]
x = prep_u8(patch).to(dev, dtype=torch.float16)
Rg = mm.Runner(MD)
with torch.no_grad():
    g_out = to_u8(Rg.infer(x.clone().float(), prompt=""))
Rn = mm.Runner(MD)
Rn.graph = None          # force eager path
with torch.no_grad():
    n_out = to_u8(Rn.infer(x.clone().float(), prompt=""))
m = torch.jit.load(os.path.join(MD, "your_model.pt"), map_location=dev); m.eval()
with torch.no_grad():
    j_out = to_u8(m(x.clone()))
print("[%s] first-tile comparisons:" % TAG, flush=True)
cmp("graph path", g_out, "no-graph path", n_out)
cmp("direct jit", j_out, "graph path", g_out)
