r"""E21: refiner + 可微 CLIP-IQA 奖励训练（clipiqa 是本会话外推能力最好的代理）。
在 L1+LPIPS+GAN 之外加 -w_clip * CLIPIQA(out)（最大化 clipiqa）。
双尺度对冲（E15 教训：裁剪尺度变好可能在全图尺度变差）：把 crop 分别以 1.0 与 0.25 缩放后送 CLIPIQA 取平均。
用法: python opt11_train_refiner_clip.py --steps 6000 --tag clip1 --ch 16 --blocks 2 --w_clip 5 --tgt_dir D:\datasets\distill256\tgt
"""
import argparse, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--w_gan", type=float, default=0.3)
ap.add_argument("--w_clip", type=float, default=5.0)
ap.add_argument("--clip_scales", default="1.0,0.25")
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--ch", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--init", default="")
ap.add_argument("--tag", default="clip1")
ap.add_argument("--tgt_dir", default=r"D:\datasets\distill256\tgt")
a = ap.parse_args()
IN = Path(r"D:\models\refine_v1\in"); TGT = Path(a.tgt_dir)
OUT = Path(r"D:\models\refine_v1"); LOG = OUT / ("train_%s.log" % a.tag)
dev = "cuda"
torch.manual_seed(0); np.random.seed(0)
SCALES = [float(x) for x in a.clip_scales.split(",")]

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
    G.load_state_dict(sd["net"]); print("[init] refiner <-", Path(a.init).name, flush=True)

from pyiqa.archs.clipiqa_arch import CLIPIQA
M = CLIPIQA(model_type="clipiqa", backbone="RN50").to(dev).eval()
for p in M.parameters(): p.requires_grad_(False)
print("[clipiqa] 可微奖励已装载 scales=%s w_clip=%.2f" % (SCALES, a.w_clip), flush=True)

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

def clip_reward(out):
    """双尺度 clipiqa（fp32，绕开 autocast）。"""
    vals = []
    with torch.autocast("cuda", enabled=False):
        o = out.float()
        for s_ in SCALES:
            if s_ == 1.0: z = o
            else:
                nh = max(64, int(round(o.shape[-2]*s_/8))*8); nw = max(64, int(round(o.shape[-1]*s_/8))*8)
                z = F.interpolate(o, size=(nh, nw), mode="bilinear", align_corners=False)
            vals.append(M(z).mean())
    return torch.stack(vals).mean()

t0 = time.time(); rg = rd = rc = 0.0
for step in range(1, a.steps + 1):
    xin, tgt = sample()
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            fake = G(xin)
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
    cr = clip_reward(out)
    lossG = lossG - a.w_clip * cr
    optG.zero_grad(set_to_none=True); scG.scale(lossG).backward(); scG.step(optG); scG.update(); schG.step()
    rg += float(lossG); rd += float(lossD); rc += float(cr)
    if step % 100 == 0:
        log("step %d/%d G=%.4f D=%.4f (l1=%.4f lp=%.4f gan=%.3f clip=%.4f) %.2f s/step" %
            (step, a.steps, rg/100, rd/100, float(l1), float(lpp), float(gan), rc/100, (time.time()-t0)/step)); rg = rd = rc = 0.0
    if step % 1000 == 0 or step == a.steps:
        torch.save({"net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step},
                   OUT / ("refiner_%s_s%d.pt" % (a.tag, step)))
log("REFINER_CLIP_DONE")
