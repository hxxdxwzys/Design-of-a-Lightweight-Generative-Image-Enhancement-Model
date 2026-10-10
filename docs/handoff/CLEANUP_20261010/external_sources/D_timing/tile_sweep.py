# -*- coding: utf-8 -*-
"""T 系列回归：4K 渲染的"分块几何"轴。
在【原生 4K 口径】下比较不同 tile/overlap/融合策略 —— 这是项目从未系统扫过的一轴
（历史只在 512 中心裁切口径下试过一次 overlap 改动）。

用法: python tile_sweep.py --tag T0 --tile 512 --over 32 [--n 8] [--imgs case3,case5,case17,...]
输出: D:\\models\\opt11\\tile_<tag>\\*.jpg  +  D:\\models\\opt11\\tile_<tag>.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
OUTROOT = Path(r"D:\models\opt11")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--tile", type=int, default=512)
ap.add_argument("--over", type=int, default=32)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--imgs", default="")
ap.add_argument("--quality", type=int, default=98)
a = ap.parse_args()

import importlib.util
spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
R = mm.Runner(MD)
dev = str(R.device)

TILE, OVER = a.tile, a.over


def make_feather(tile, over):
    m = np.ones((tile, tile, 1), dtype=np.float32)
    r = over // 2
    if r > 0:
        ramp = np.linspace(0.0, 1.0, r, dtype=np.float32)
        m[:r, :, 0] *= ramp[:, None]
        m[-r:, :, 0] *= ramp[::-1, None]
        m[:, :r, 0] *= ramp[None, :]
        m[:, -r:, 0] *= ramp[::-1][None, :]
    return m


MASK = make_feather(TILE, OVER)


def tile_grid(H, W, t, o):
    ys = sorted(set(list(range(0, max(1, H - t + 1), max(1, t - o))) + [max(0, H - t)]))
    xs = sorted(set(list(range(0, max(1, W - t + 1), max(1, t - o))) + [max(0, W - t)]))
    return ys, xs


@torch.no_grad()
def render(lq):
    H, W = lq.shape[:2]
    t = min(TILE, H, W)
    mk = MASK if t == TILE else np.ones((t, t, 1), dtype=np.float32)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys, xs = tile_grid(H, W, t, OVER)
    for y0 in ys:
        for x0 in xs:
            patch = lq[y0:y0 + t, x0:x0 + t]
            x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
            y = R.infer(x, prompt="")
            out = y[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
            acc[y0:y0 + t, x0:x0 + t] += np.clip((out + 1.0) * 127.5, 0, 255) * mk
            wsum[y0:y0 + t, x0:x0 + t] += mk
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8), len(ys) * len(xs)


if a.imgs:
    files = [TEST / (s.strip() + ".jpg") for s in a.imgs.split(",") if s.strip()]
else:
    files = sorted(TEST.glob("case*.jpg"))[:a.n]

outdir = OUTROOT / ("tile_%s" % a.tag)
outdir.mkdir(parents=True, exist_ok=True)
print("[tile_sweep] tag=%s tile=%d over=%d  imgs=%d -> %s" % (a.tag, TILE, OVER, len(files), outdir), flush=True)

rec = []
t_all = time.time()
for i, f in enumerate(files, 1):
    lq = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    H, W = lq.shape[:2]
    ys, xs = tile_grid(H, W, min(TILE, H, W), OVER)
    t0 = time.time()
    res, nt = render(lq)
    dt = time.time() - t0
    Image.fromarray(res).save(outdir / f.name, quality=a.quality, subsampling=0)
    rec.append({"case": f.stem, "H": H, "W": W, "tiles": nt, "sec": round(dt, 2)})
    print("  %2d/%d %-9s %dx%d tiles=%-3d %6.2f s" % (i, len(files), f.stem, W, H, nt, dt), flush=True)

meta = {"tag": a.tag, "tile": TILE, "over": OVER, "n": len(files), "dir": str(outdir),
        "total_sec": round(time.time() - t_all, 1), "per_image": rec}
(OUTROOT / ("tile_%s.json" % a.tag)).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
print("[tile_sweep] DONE %s  total=%.1f s  avg=%.1f s/张" % (a.tag, meta["total_sec"], meta["total_sec"] / len(files)))
print("TILE_SWEEP_DONE", a.tag)
