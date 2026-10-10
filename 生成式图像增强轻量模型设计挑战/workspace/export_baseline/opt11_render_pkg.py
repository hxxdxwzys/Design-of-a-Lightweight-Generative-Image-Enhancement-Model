r"""用【打包后的 runner 本身】渲染测试集 4K（512 块, overlap 32, 羽化, q98）。
用法: python opt11_render_pkg.py --modeldir DIR --out DIR [--n 30 --order lex]
"""
import argparse, importlib.util, os, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
TILE, OVER = 512, 32
ap = argparse.ArgumentParser()
ap.add_argument("--modeldir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--order", default="lex")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("pkg_runner", os.path.join(a.modeldir, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(a.modeldir)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]

@torch.no_grad()
def run(patch_u8):
    t = torch.from_numpy(patch_u8).permute(2, 0, 1)[None].float().div_(127.5).sub_(1)
    y = R.infer(t, prompt="")
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)

files = sorted(TEST.glob("case*.jpg")) if a.order == "lex" else sorted(TEST.glob("case*.jpg"), key=lambda p: int(p.stem[4:]))
if a.n: files = files[:a.n]
print("[render_pkg] %d 张 modeldir=%s" % (len(files), a.modeldir), flush=True)
t0 = time.time()
for i, f in enumerate(files, 1):
    img = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H-TILE+1), TILE-OVER)) + [max(0, H-TILE)]))
    xs_ = sorted(set(list(range(0, max(1, W-TILE+1), TILE-OVER)) + [max(0, W-TILE)]))
    for y0 in ys:
        for x0 in xs_:
            acc[y0:y0+TILE, x0:x0+TILE] += run(img[y0:y0+TILE, x0:x0+TILE]) * mask
            wsum[y0:y0+TILE, x0:x0+TILE] += mask
    res = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(res).save(out / f.name, quality=98, subsampling=0)
    if i % 10 == 0 or i == len(files):
        print("  %d/%d %.2f s/张" % (i, len(files), (time.time()-t0)/i), flush=True)
print("RENDER_PKG_DONE", flush=True)
