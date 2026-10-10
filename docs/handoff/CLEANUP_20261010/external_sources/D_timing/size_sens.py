# -*- coding: utf-8 -*-
"""检查模型对输入尺寸的敏感性与大块/dense 前向的可行性。
输出：同一区域在不同送入尺寸下的预测差异 + 显存占用 + 单次耗时。
"""
import os, sys, importlib.util, time
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
dev = "cuda"

spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)


def infer_u8(patch):
    x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
    with torch.no_grad():
        y = R.infer(x, prompt="")
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


img = np.asarray(Image.open(os.path.join(TEST, "case3.jpg")).convert("RGB"), dtype=np.uint8)
region = img[512:1536, 512:1536]           # 1024x1024 区域
print("region", region.shape)

# 1) dense 1024 一次前向 可行性 + 耗时 + 显存
torch.cuda.reset_peak_memory_stats()
try:
    t = time.time()
    dense = infer_u8(region)
    torch.cuda.synchronize()
    print("[dense 1024] OK  %.2f s  peak mem %.0f MiB" % (time.time() - t, torch.cuda.max_memory_allocated() / 2**20), flush=True)
except Exception as e:
    print("[dense 1024] FAILED:", str(e)[:200]); dense = None

# 2) 同一 512 子块，在不同送入尺寸下（真值上下文）取同一坐标的输出
sub_y, sub_x = 256, 256                     # dense 坐标内的子块起点
for size in (512, 768, 1024):
    if size > region.shape[0]:
        continue
    y0 = min(max(0, sub_y - (size - 512) // 2), region.shape[0] - size)
    x0 = min(max(0, sub_x - (size - 512) // 2), region.shape[1] - size)
    big = region[y0:y0 + size, x0:x0 + size]
    out = infer_u8(big)
    sub = out[sub_y - y0: sub_y - y0 + 512, sub_x - x0: sub_x - x0 + 512]
    if dense is not None:
        ref = dense[sub_y:sub_y + 512, sub_x:sub_x + 512].astype(np.int16)
        d = np.abs(sub.astype(np.int16) - ref)
        print("[send %4d] 同一512子块 vs dense(1024直送): max=%3d mean=%.3f pct!=0=%.1f%%"
              % (size, d.max(), d.mean(), (d != 0).mean() * 100), flush=True)
    else:
        print("[send %4d] OK" % size, flush=True)
    del out, big

# 3) 更大尺寸可行性（显存边界）
for size in (1536, 2048):
    try:
        torch.cuda.reset_peak_memory_stats()
        big = img[0:size, 0:size] if img.shape[0] >= size and img.shape[1] >= size else None
        if big is None:
            print("[probe %4d] 图像不够大，跳过" % size); continue
        t = time.time()
        out = infer_u8(big)
        torch.cuda.synchronize()
        print("[probe %4d] OK  %.2f s  peak mem %.0f MiB" % (size, time.time() - t, torch.cuda.max_memory_allocated() / 2**20), flush=True)
        del out, big
    except Exception as e:
        print("[probe %4d] FAILED: %s" % (size, str(e)[:150]), flush=True)
