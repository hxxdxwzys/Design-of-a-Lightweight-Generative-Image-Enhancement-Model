r"""E9 渲染侧实验：外部拼接 overlap / 羽化窗 对质量代理的影响（与测速无关，不计时）。
用法: python opt11_e9_overlap.py <over> <window:linear|hann> <tag>
"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
over = int(sys.argv[1]); window = sys.argv[2]; tag = sys.argv[3]
TILE = 512
MD = r"D:\models\pkg_TinyRefV4\model_dir"
spec = importlib.util.spec_from_file_location("e9r", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
out = Path(r"D:\models\opt11\e9_%s" % tag); out.mkdir(parents=True, exist_ok=True)
r = over // 2
ramp = np.linspace(0, 1, r, dtype=np.float32) if r else np.array([], dtype=np.float32)
if window == "hann" and r:
    ramp = (0.5 - 0.5*np.cos(np.pi*ramp)).astype(np.float32)
mask = np.ones((TILE, TILE, 1), dtype=np.float32)
if r:
    mask[:r,:,0] *= ramp[:,None]; mask[-r:,:,0] *= ramp[::-1,None]
    mask[:,:r,0] *= ramp[None,:]; mask[:,-r:,0] *= ramp[::-1][None,:]
@torch.no_grad()
def tile(patch):
    t = torch.from_numpy(patch.copy()).permute(2,0,1)[None].float().div_(127.5).sub_(1)
    y = R.infer(t, prompt="")
    return np.clip((y[0].permute(1,2,0).float().cpu().numpy()+1)*127.5, 0, 255)
files = sorted(TEST.glob("case*.jpg"))[:30]
t0 = time.time()
for i, f in enumerate(files, 1):
    img = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H,W,3), dtype=np.float32); ws = np.zeros((H,W,1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1,H-TILE+1), TILE-over)) + [max(0,H-TILE)]))
    xs = sorted(set(list(range(0, max(1,W-TILE+1), TILE-over)) + [max(0,W-TILE)]))
    for y0 in ys:
        for x0 in xs:
            acc[y0:y0+TILE, x0:x0+TILE] += tile(img[y0:y0+TILE, x0:x0+TILE]) * mask
            ws[y0:y0+TILE, x0:x0+TILE] += mask
    res = np.clip(acc/np.maximum(ws,1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(res).save(out / f.name, quality=98, subsampling=0)
print("[E9] over=%d window=%s -> %s (%.1f min)" % (over, window, out, (time.time()-t0)/60), flush=True)
print("E9_RENDER_DONE", flush=True)
