r"""r128 refiner：128 域两级像素重排 -> 512（学生与学生输出都在 128 域，算力最省）。
用法: python opt11_train_refiner128.py --steps 12000 --tag r128a [--ch 32 --blocks 2 --crop 64]
"""
import argparse, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_d", type=float, default=1.5e-4)
ap.add_argument("--w_lpips", type=float, default=0.2); ap.add_argument("--w_gan", type=float, default=0.12)
ap.add_argument("--crop", type=int, default=64); ap.add_argument("--ch", type=int, default=32)
ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--tag", default="r128a")
ap.add_argument("--init", default="")
a = ap.parse_args()
IN = Path(r"D:\models\refine_v1\in128"); TGT = Path(r"D:\datasets\distill256\tgt")
OUT = Path(r"D:\models\refine_v1"); LOG = OUT / ("train_%s.log" % a.tag)
dev = "cuda"; torch.manual_seed(0); np.random.seed(0)

class Refiner128(nn.Module):
    def __init__(self, ch=32, blocks=2):
        super().__init__()
        L = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): L += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        L += [nn.Conv2d(ch, 12, 3, 1, 1)]
        self.net = nn.Sequential(*L)
        self.mid = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True), nn.Conv2d(ch, 12, 3, 1, 1))
        self.ps = nn.PixelShuffle(2)
    def forward(self, x):                     # (B,3,128,128) -> (B,3,512,512)
        u1 = self.ps(self.net(x)) + F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        u2 = self.ps(self.mid(u1))
        base = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
        return (base + u2).clamp(-1, 1)

class PatchD(nn.Module):
    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev = mult; mult = min(2**i, 8)
            layers += [nn.Conv2d(ch*prev, ch*mult, 4, 2, 1, bias=False), nn.BatchNorm2d(ch*mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch*mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

G = Refiner128(a.ch, a.blocks).to(dev); D = PatchD().to(dev)
if a.init and Path(a.init).exists():
    sd = torch.load(a.init, map_location="cpu", weights_only=False); G.load_state_dict(sd["net"], strict=False); print("[init]", Path(a.init).name, flush=True)
print("[model] r128 ch=%d blocks=%d params=%.3fM" % (a.ch, a.blocks, sum(p.numel() for p in G.parameters())/1e6), flush=True)
import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)
optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9,0.99)); optD = torch.optim.AdamW(D.parameters(), lr=a.lr_d, betas=(0.9,0.99))
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=a.steps, eta_min=a.lr*0.05)
scG = torch.amp.GradScaler("cuda"); scD = torch.amp.GradScaler("cuda")
files = sorted(IN.glob("*.png")); print("[data]", len(files), flush=True)
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
def sample():
    ins, tg = [], []
    c = a.crop
    for _ in range(a.bs):
        f = files[np.random.randint(len(files))]
        i = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        g = np.asarray(Image.open(TGT / (f.stem + ".jpg")).convert("RGB"), dtype=np.uint8)
        y0 = np.random.randint(0, i.shape[0]-c+1); x0 = np.random.randint(0, i.shape[1]-c+1)
        ins.append(i[y0:y0+c, x0:x0+c]); tg.append(g[4*y0:4*(y0+c), 4*x0:4*(x0+c)])
    to = lambda arr: torch.from_numpy(np.stack(arr)).permute(0,3,1,2).to(dev).float().div_(127.5).sub_(1)
    return to(ins), to(tg)
t0 = time.time(); rg = rd = 0.0
for step in range(1, a.steps+1):
    xin, tgt = sample()
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad(): fake = G(xin)
        pr, pf = D(tgt), D(fake)
        lossD = 0.5*(F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
    optD.zero_grad(set_to_none=True); scD.scale(lossD).backward(); scD.step(optD); scD.update()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(xin)
        l1 = F.l1_loss(out.float(), tgt); lpp = lp(out.float(), tgt).mean()
        pf2 = D(out); gan = F.mse_loss(pf2, torch.ones_like(pf2))
        lossG = l1 + a.w_lpips*lpp + a.w_gan*gan
    optG.zero_grad(set_to_none=True); scG.scale(lossG).backward(); scG.step(optG); scG.update(); schG.step()
    rg += float(lossG); rd += float(lossD)
    if step % 200 == 0:
        log("step %d/%d G=%.4f D=%.4f (l1=%.4f lp=%.4f gan=%.3f) %.2f s/step" % (step, a.steps, rg/200, rd/200, float(l1), float(lpp), float(gan), (time.time()-t0)/step)); rg = rd = 0.0
    if step % 3000 == 0 or step == a.steps:
        torch.save({"net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step, "arch": "r128"}, OUT / ("refiner_%s_s%d.pt" % (a.tag, step)))
log("REFINER128_DONE")
