r"""细节合成器 (refiner) 训练: 输入=学生 512 域输出, 目标=M24 观感 (distill256/tgt)。
用法: python opt11_train_refiner.py --steps 3000 --tag pilot [--ch 48 --blocks 3 --bs 8]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--ch", type=int, default=48)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--tag", default="pilot")
a = ap.parse_args()
IN = Path(r"D:\models\refine_v1\in"); TGT = Path(r"D:\datasets\distill256\tgt")
OUT = Path(r"D:\models\refine_v1"); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

class Refiner(nn.Module):
    def __init__(self, ch=48, blocks=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return (x + self.net(x)).clamp(-1, 1)

G = Refiner(a.ch, a.blocks).to(dev)
print("refiner params = %.3fM" % (sum(p.numel() for p in G.parameters())/1e6), flush=True)
import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr*0.05)
scaler = torch.amp.GradScaler("cuda")
files = sorted(IN.glob("*.jpg")); print("[data]", len(files), "对", flush=True)

def sample():
    ins, tgts = [], []
    for _ in range(a.bs):
        k = np.random.randint(len(files))
        f = files[k]
        t = TGT / (f.stem + ".jpg")
        i = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        g = np.asarray(Image.open(t).convert("RGB"), dtype=np.uint8)
        if i.shape != g.shape:  # 兜底
            g = np.asarray(Image.open(t).convert("RGB").resize((i.shape[1], i.shape[0])), dtype=np.uint8)
        c = a.crop
        y0 = np.random.randint(0, i.shape[0]-c+1); x0 = np.random.randint(0, i.shape[1]-c+1)
        ins.append(i[y0:y0+c, x0:x0+c]); tgts.append(g[y0:y0+c, x0:x0+c])
    to = lambda arr: torch.from_numpy(np.stack(arr)).permute(0,3,1,2).to(dev).float().div_(127.5).sub_(1)
    return to(ins), to(tgts)

LOG = OUT / ("train_%s.log" % a.tag)
def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")
t0 = time.time(); run = 0.0
for step in range(1, a.steps+1):
    xin, tgt = sample()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(xin)
        l1 = F.l1_loss(out.float(), tgt)
        lpp = lp(out.float()*2-1, tgt*2-1).mean() if False else lp(out.float(), tgt).mean()
        loss = l1 + a.w_lpips * lpp
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    run += float(loss)
    if step % 100 == 0:
        log("step %d/%d loss=%.4f (l1=%.4f lp=%.4f) %.2f s/step" %
            (step, a.steps, run/100, float(l1), float(lpp), (time.time()-t0)/step)); run = 0.0
    if step % 1000 == 0 or step == a.steps:
        torch.save({"net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step},
                   OUT / ("refiner_%s_s%d.pt" % (a.tag, step)))
log("REFINER_TRAIN_DONE")
