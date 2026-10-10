r"""细节合成器 + 对抗纹理 (PatchGAN) 训练，从 L1+LPIPS 版初始化。
用法: python opt11_train_refiner_gan.py --steps 10000 --tag gan1 [--init D:\models\refine_v1\refiner_v1_s20000.pt]
"""
import argparse, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--w_gan", type=float, default=0.1)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--ch", type=int, default=48)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--init", default="")
ap.add_argument("--tag", default="gan1")
ap.add_argument("--tgt_dir", default=r"D:\datasets\distill256\tgt",
                help="训练目标目录；E8 用 VOSR 观感目标 D:/models/refine_v1/tgt_vosr")
a = ap.parse_args()
IN = Path(r"D:\models\refine_v1\in"); TGT = Path(a.tgt_dir)
OUT = Path(r"D:\models\refine_v1"); LOG = OUT / ("train_%s.log" % a.tag)
dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

class Refiner(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return (x + self.net(x)).clamp(-1, 1)

class PatchD(nn.Module):
    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev = mult; mult = min(2 ** i, 8)
            layers += [nn.Conv2d(ch * prev, ch * mult, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

G = Refiner(a.ch, a.blocks).to(dev)
D = PatchD().to(dev)
if a.init and Path(a.init).exists():
    sd = torch.load(a.init, map_location="cpu", weights_only=False)
    G.load_state_dict(sd["net"]); print("[init] refiner <-", Path(a.init).name, "steps", sd.get("steps"), flush=True)
import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)
optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
optD = torch.optim.AdamW(D.parameters(), lr=a.lr_d, betas=(0.9, 0.99))
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=a.steps, eta_min=a.lr * 0.05)
scG = torch.amp.GradScaler("cuda"); scD = torch.amp.GradScaler("cuda")
files = sorted(IN.glob("*.jpg")); print("[data]", len(files), flush=True)

def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")

def sample():
    ins, tg = [], []
    for _ in range(a.bs):
        f = files[np.random.randint(len(files))]
        i = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        g = np.asarray(Image.open(TGT / (f.stem + ".jpg")).convert("RGB"), dtype=np.uint8)
        c = a.crop; y0 = np.random.randint(0, i.shape[0]-c+1); x0 = np.random.randint(0, i.shape[1]-c+1)
        ins.append(i[y0:y0+c, x0:x0+c]); tg.append(g[y0:y0+c, x0:x0+c])
    to = lambda arr: torch.from_numpy(np.stack(arr)).permute(0,3,1,2).to(dev).float().div_(127.5).sub_(1)
    return to(ins), to(tg)

t0 = time.time(); rg = rd = 0.0
for step in range(1, a.steps + 1):
    xin, tgt = sample()
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            fake = G(xin)                     # D 复用，不反传 G
        pred_real = D(tgt); pred_fake = D(fake)
        lossD = 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                       F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
    optD.zero_grad(set_to_none=True); scD.scale(lossD).backward(); scD.step(optD); scD.update()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(xin)
        l1 = F.l1_loss(out.float(), tgt)
        lpp = lp(out.float(), tgt).mean()
        pf = D(out)
        gan = F.mse_loss(pf, torch.ones_like(pf))
        lossG = l1 + a.w_lpips * lpp + a.w_gan * gan
    optG.zero_grad(set_to_none=True); scG.scale(lossG).backward(); scG.step(optG); scG.update(); schG.step()
    rg += float(lossG); rd += float(lossD)
    if step % 100 == 0:
        log("step %d/%d G=%.4f D=%.4f (l1=%.4f lp=%.4f gan=%.3f) %.2f s/step" %
            (step, a.steps, rg/100, rd/100, float(l1), float(lpp), float(gan), (time.time()-t0)/step)); rg = rd = 0.0
    if step % 2000 == 0 or step == a.steps:
        torch.save({"net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step},
                   OUT / ("refiner_%s_s%d.pt" % (a.tag, step)))
log("REFINER_GAN_DONE")
