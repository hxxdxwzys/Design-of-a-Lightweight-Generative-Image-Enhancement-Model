r"""E11 诊断：把 refiner 直接作用在 VOSR(OPT-10) 的 4K 出图上（逐 512 块 + 羽化），测其质量代理。
判读：若 refiner 保真 ⇒ 输出 musiq ≈ 输入(65.9)；若破坏 ⇒ 显著下降。决定瓶颈在学生还是 refiner。
用法: python opt11_e11_refiner_on_vosr.py <refiner_ckpt> <tag>
"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ref_ckpt = sys.argv[1]; tag = sys.argv[2]
SRC = Path(r"D:\models\opt10\output_dir"); OUT = Path(r"D:\models\opt11\e11_%s" % tag); OUT.mkdir(parents=True, exist_ok=True)
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
dev = "cuda"; TILE, OVER = 512, 32

class Refiner(nn.Module):
    def __init__(self, ch, blocks, arch="default"):
        super().__init__()
        self.arch = arch
        out_ch = 12 if arch == "r256" else 3
        L = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): L += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        L += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*L); self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        if self.arch == "r256":
            base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            return (base + self.ps(self.net(x))).clamp(-1, 1)
        return (x + self.net(x)).clamp(-1, 1)

rd = torch.load(ref_ckpt, map_location="cpu", weights_only=False)
ref = Refiner(rd["ch"], rd["blocks"], rd.get("arch","default")).to(dev).eval().half(); ref.load_state_dict(rd["net"])
print("[E11] refiner ch=%d blocks=%d arch=%s steps=%s" % (rd["ch"], rd["blocks"], rd.get("arch","default"), rd.get("steps")), flush=True)
mask = np.ones((TILE,TILE,1), dtype=np.float32); r = OVER//2
ramp = np.linspace(0,1,r,dtype=np.float32)
mask[:r,:,0] *= ramp[:,None]; mask[-r:,:,0] *= ramp[::-1,None]
mask[:,:r,0] *= ramp[None,:]; mask[:,-r:,0] *= ramp[::-1][None,:]
names = [f.stem for f in sorted(TEST.glob("*.jpg"))[:30]]
t0 = time.time()
with torch.no_grad():
    for i, nm in enumerate(names, 1):
        img = np.asarray(Image.open(SRC / ("%s.jpg" % nm)).convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]
        acc = np.zeros((H,W,3), dtype=np.float32); ws = np.zeros((H,W,1), dtype=np.float32)
        ys = sorted(set(list(range(0, max(1,H-TILE+1), TILE-OVER)) + [max(0,H-TILE)]))
        xs = sorted(set(list(range(0, max(1,W-TILE+1), TILE-OVER)) + [max(0,W-TILE)]))
        for y0 in ys:
            for x0 in xs:
                p = img[y0:y0+TILE, x0:x0+TILE]
                t = torch.from_numpy(p.copy()).permute(2,0,1)[None].to(dev).float().div_(127.5).sub_(1).half()
                o = ref(t)
                arr = ((o[0].permute(1,2,0).float().cpu().numpy()+1)*127.5)
                acc[y0:y0+TILE, x0:x0+TILE] += arr * mask; ws[y0:y0+TILE, x0:x0+TILE] += mask
        res = np.clip(acc/np.maximum(ws,1e-6), 0, 255).astype(np.uint8)
        Image.fromarray(res).save(OUT / ("%s.jpg" % nm), quality=98, subsampling=0)
        if i % 10 == 0: print("  %d/30 %.1f s/张" % (i, (time.time()-t0)/i), flush=True)
print("E11_RENDER_DONE ->", OUT, flush=True)
