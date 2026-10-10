# -*- coding: utf-8 -*-
"""A2：4K 级跨块一致性。
问题：出图是 512 分块独立预测 + 32px 线性羽化拼接；实测重叠带内两邻块预测差异 mean 4.0–7.7
      （与该图自身像素级梯度同量级）。但所有后处理都是【逐块】做的，从未在【整张 4K】层面利用。

本脚本做两件事（都很便宜）：
  ① 接缝能量指标：在拼接结果上，看"分块边界处"的局部高频是否系统性地高于非边界处。
  ② 便宜地比较几种拼接配置（overlap 大小 + 混合函数），只对胜者再跑完整指标。

判据（预登记）：
  若某配置的接缝能量比现役(512/32/linear)降低 >= 30%，才值得跑完整 clipiqa/niqe。
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
OUTR = Path(r"D:\models\opt11")
TILE = 512

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--over", type=int, default=32)
ap.add_argument("--blend", default="linear", choices=["linear", "smooth", "hard_center"])
ap.add_argument("--imgs", default="case1,case2,case3,case5,case7")
a = ap.parse_args()

import importlib.util
spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
R = mm.Runner(MD); dev = str(R.device)
OVER = a.over


def make_mask(tile, over, kind):
    m = np.ones((tile, tile, 1), dtype=np.float32)
    if kind == "hard_center":
        r = over // 2
        if r > 0:
            m[:r, :, 0] = 0.0; m[-r:, :, 0] = 0.0
            m[:, :r, 0] = 0.0; m[:, -r:, 0] = 0.0
        return m
    r = over // 2
    if r > 0:
        ramp = np.linspace(1e-3, 1.0, r, dtype=np.float32) if kind == "linear" else \
               (0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, r, dtype=np.float32)))
        m[:r, :, 0] *= ramp[:, None]
        m[-r:, :, 0] *= ramp[::-1, None]
        m[:, :r, 0] *= ramp[None, :]
        m[:, -r:, 0] *= ramp[::-1][None, :]
    return m


MASK = make_mask(TILE, OVER, a.blend)


@torch.no_grad()
def render(lq):
    H, W = lq.shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H - TILE + 1), max(1, TILE - OVER))) + [max(0, H - TILE)]))
    xs = sorted(set(list(range(0, max(1, W - TILE + 1), max(1, TILE - OVER))) + [max(0, W - TILE)]))
    for y0 in ys:
        for x0 in xs:
            patch = lq[y0:y0 + TILE, x0:x0 + TILE]
            x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
            y = R.infer(x, prompt="")
            o = y[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
            acc[y0:y0 + TILE, x0:x0 + TILE] += np.clip((o + 1.0) * 127.5, 0, 255) * MASK
            wsum[y0:y0 + TILE, x0:x0 + TILE] += MASK
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8), ys, xs


def seam_energy(img, ys, xs):
    """接缝能量：分块边界处的相邻列/行差分，与非边界处的比值。>1 表示边界处有系统性不连续。"""
    g = img.astype(np.float32).mean(2)
    dx = np.abs(np.diff(g, axis=1))            # (H, W-1)：列 x 与 x+1 的差
    dy = np.abs(np.diff(g, axis=0))
    # 垂直边界的 x 位置集合（块左沿）
    bx = set()
    for x0 in xs:
        for e in (x0, x0 + TILE - 1):
            if 1 <= e < g.shape[1] - 1:
                bx.add(e)
    by = set()
    for y0 in ys:
        for e in (y0, y0 + TILE - 1):
            if 1 <= e < g.shape[0] - 1:
                by.add(e)
    if not bx or not by:
        return None
    onx = np.array(sorted(bx)); ony = np.array(sorted(by))
    col = dx.mean(0)                            # 每列的列间差均值
    row = dy.mean(1)
    se_x = float(col[onx].mean() / col.mean())
    se_y = float(row[ony].mean() / row.mean())
    return {"seam_ratio_x": round(se_x, 4), "seam_ratio_y": round(se_y, 4),
            "seam_ratio": round((se_x + se_y) / 2, 4), "n_seam_x": len(onx), "n_seam_y": len(ony)}


files = [TEST / (s.strip() + ".jpg") for s in a.imgs.split(",") if s.strip()]
outdir = OUTR / ("a2_%s" % a.tag); outdir.mkdir(parents=True, exist_ok=True)
print("[a2] tag=%s over=%d blend=%s imgs=%d" % (a.tag, OVER, a.blend, len(files)), flush=True)
recs = []
for f in files:
    lq = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    res, ys, xs = render(lq)
    Image.fromarray(res).save(outdir / f.name, quality=98, subsampling=0)
    se = seam_energy(res, ys, xs)
    recs.append({"case": f.stem, **se})
    print("   %-8s seam_ratio=%.4f (x=%.4f y=%.4f)" % (f.stem, se["seam_ratio"], se["seam_ratio_x"], se["seam_ratio_y"]), flush=True)

agg = {"tag": a.tag, "over": OVER, "blend": a.blend, "n": len(files),
       "seam_ratio_mean": round(float(np.mean([r["seam_ratio"] for r in recs])), 4),
       "per_image": recs}
(OUTR / ("a2_%s.json" % a.tag)).write_text(json.dumps(agg, indent=1, ensure_ascii=False), encoding="utf-8")
print("[a2] %s -> seam_ratio_mean = %.4f" % (a.tag, agg["seam_ratio_mean"]), flush=True)
print("A2_SEAM_DONE", a.tag)
