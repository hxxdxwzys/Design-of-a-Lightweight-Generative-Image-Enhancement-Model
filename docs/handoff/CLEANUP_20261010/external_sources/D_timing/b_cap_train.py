# -*- coding: utf-8 -*-
"""B1'（量级跃迁·容量做对版）：大容量 refiner，充分训练。

为什么这条值得重做（而不是重走 E19）：
  E19 当年得出"容量无效"，但那是 **6k 步 + w_gan 0.3** 的欠训+对抗过强配置；
  本会话已实测「训练步数比容量关键」（thin2 续训 6k 步比新训 16ch 高 1.6%）。
  => **从未在【充分训练】下测过大容量 refiner。**

单变量：refiner 形状 16ch/2bl(11K) -> 64ch/4bl(~234K, 21x)；数据/目标/损失/unsharp 全同。

判据（预登记）：
  全链 clipiqa > 0.8118  => 净胜最优（V4t1）
  > 0.8042               => 优于已提交 V4
  <= 0.8042              => 容量轴在充分训练下也 CLOSED
外加：必须记录 512 时延（大 refiner 更慢，须扣分判净值）。

用法: python b_cap_train.py --ch 64 --blocks 4 --steps 20000 --tag cap64
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cap64")
ap.add_argument("--ch", type=int, default=64)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--w_gan", type=float, default=0.1)
ap.add_argument("--crop", type=int, default=256)
a = ap.parse_args()

dev = "cuda"
IN = Path(r"D:\models\refine_v1\in")
TGT = Path(r"D:\datasets\distill256\tgt")
OUT = Path(r"D:\models\refine_v1")
LOG = OUT / ("train_%s.log" % a.tag)
torch.manual_seed(0); np.random.seed(0)


class Refiner(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x + self.net(x)).clamp(-1, 1)


class PatchD(nn.Module):
    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        prev = ch
        for i in range(n_layers):
            mult = min(2 ** i, 8)
            layers += [nn.Conv2d(prev, ch * mult, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch * mult), nn.LeakyReLU(0.2, True)]
            prev = ch * mult
        layers += [nn.Conv2d(prev, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def log(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m); print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


G = Refiner(a.ch, a.blocks).to(dev)
D = PatchD().to(dev)
log("[model] G=%d params (ch=%d blocks=%d)  D=%d params"
    % (sum(p.numel() for p in G.parameters()), a.ch, a.blocks, sum(p.numel() for p in D.parameters())))

import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters():
    p.requires_grad_(False)
optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
optD = torch.optim.AdamW(D.parameters(), lr=a.lr_d, betas=(0.9, 0.99))
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=a.steps, eta_min=a.lr * 0.05)
schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=a.steps, eta_min=a.lr_d * 0.05)
scG = torch.amp.GradScaler("cuda"); scD = torch.amp.GradScaler("cuda")
files = sorted(IN.glob("*.jpg"))
log("[data] %d 对（随机 %d 裁剪）" % (len(files), a.crop))


def sample():
    ins, tg = [], []
    for _ in range(a.bs):
        f = files[np.random.randint(len(files))]
        i = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        g = np.asarray(Image.open(TGT / (f.stem + ".jpg")).convert("RGB"), dtype=np.uint8)
        c = a.crop
        y0 = np.random.randint(0, i.shape[0] - c + 1); x0 = np.random.randint(0, i.shape[1] - c + 1)
        ins.append(i[y0:y0 + c, x0:x0 + c]); tg.append(g[y0:y0 + c, x0:x0 + c])
    to = lambda arr: torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)
    return to(ins), to(tg)


t0 = time.time(); rg = rd = 0.0
for step in range(1, a.steps + 1):
    xin, tgt = sample()
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            fake = G(xin)
        pr = D(tgt); pf = D(fake)
        lossD = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
    optD.zero_grad(set_to_none=True); scD.scale(lossD).backward(); scD.step(optD); scD.update()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(xin)
        l1 = F.l1_loss(out.float(), tgt)
        lpp = lp(out.float(), tgt).mean()
        pfo = D(out)
        gan = F.mse_loss(pfo, torch.ones_like(pfo))
        lossG = l1 + a.w_lpips * lpp + a.w_gan * gan
    optG.zero_grad(set_to_none=True); scG.scale(lossG).backward(); scG.step(optG); scG.update(); schG.step()
    rg += float(lossG); rd += float(lossD)
    if step % 100 == 0:
        log("step %d/%d G=%.4f D=%.4f (l1=%.4f lp=%.4f gan=%.3f) %.2f s/step"
            % (step, a.steps, rg / 100, rd / 100, float(l1), float(lpp), float(gan), (time.time() - t0) / step))
        rg = rd = 0.0
    if step % 2000 == 0 or step == a.steps:
        torch.save({"net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step},
                   OUT / ("bcap_%s_s%d.pt" % (a.tag, step)))
        log("[save] bcap_%s_s%d.pt" % (a.tag, step))
log("BCAP_TRAIN_DONE")
