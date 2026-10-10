# -*- coding: utf-8 -*-
"""接缝诊断：现有 V4 渲染（tile=512, overlap=32）在分块边界的重叠区内部一致性如何？
若两个邻块在重叠带内的预测本身差异大，就说明"独立预测 + 羽化"是质量损失源。
"""
import os, sys, importlib.util
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
TILE, OVER = 512, 32
dev = "cuda"

spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)


def to_u8(y):
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


def run_guess_free(patch):
    """单块推理：与在线出图路径完全一致（不额外补上下文）。"""
    x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
    with torch.no_grad():
        y = R.infer(x, prompt="")
    return to_u8(y)


def run_with_context(patch, image, y0, x0, ctx=128):
    """带上下文补丁：把 patch 周围的真实像素一并送入，只取中心结果。"""
    H, W = image.shape[:2]
    ya, yb = max(0, y0 - ctx), min(H, y0 + TILE + ctx)
    xa, xb = max(0, x0 - ctx), min(W, x0 + TILE + ctx)
    big = image[ya:yb, xa:xb]
    out = run_guess_free(big)
    return out[y0 - ya: y0 - ya + TILE, x0 - xa: x0 - xa + TILE]


for nm in ("case3", "case17", "case5"):
    img = np.asarray(Image.open(os.path.join(TEST, nm + ".jpg")).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    print("\n== %s  (%dx%d) ==" % (nm, W, H), flush=True)

    # --- 接缝一致性：同一坐标，块 A（不含右邻） vs 块 B（跨接缝的下一块） ---
    y0, x0 = 512, 512          # 第一行第二列的块
    a = run_guess_free(img[y0:y0 + TILE, x0:x0 + TILE])
    # 水平方向：重叠带 = x0+TILE-OVER .. x0+TILE
    b = run_guess_free(img[y0:y0 + TILE, x0 + (TILE - OVER):x0 + TILE + (TILE - OVER)][:, :TILE])
    ov_a = a[:, TILE - OVER:]
    ov_b = b[:, :OVER]
    d = np.abs(ov_a.astype(np.int16) - ov_b.astype(np.int16))
    print("  水平接缝重叠带(32px):  max=%3d mean=%.3f  单块A内部梯度 mean|x-grad|=%.3f"
          % (d.max(), d.mean(), float(np.abs(np.diff(ov_a.astype(np.float32), axis=1)).mean())), flush=True)

    # --- 上下文补丁 vs 无上下文 ---
    for ctx in (64, 128):
        c = run_with_context(img, img, y0, x0, ctx)
        dc = np.abs(a.astype(np.int16) - c.astype(np.int16))
        # 只在块的边缘 32px 带里看差异（中心应该几乎相同）
        edge = np.zeros_like(dc, dtype=bool)
        edge[:OVER, :] = edge[-OVER:, :] = True
        edge[:, :OVER] = edge[:, -OVER:] = True
        print("  块(512,512) 有 ctx=%3d vs 无 ctx: 全块 max=%3d mean=%.3f | 边缘32px带 max=%3d mean=%.3f"
              % (ctx, dc.max(), dc.mean(), dc[edge].max(), dc[edge].mean()), flush=True)
    del img
