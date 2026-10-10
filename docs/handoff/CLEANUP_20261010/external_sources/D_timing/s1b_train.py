# -*- coding: utf-8 -*-
"""S1b（修正版）：可学习恒等细化器 U 插在【学生输出】与【refiner】之间。

设计要点（与前一版的关键差别）
  前一版：U 把 S/2 域上采样到 S 域 —— refiner 吃的是"经过二次上下采样"的输入，
          与已提交 V4（refiner 直接吃学生输出）**不可比**，且 U 受 11K 容量限制学不到恒等。
  本版  ：U 的输入输出**都在 S 域**，且 U(x) = x + zero_init_residual(x)。
          ⇒ **训练起点精确等于已提交 V4**（U 恒等、G 若也初始化为 thin2 则完全等价）。
          ⇒ 训练**只可能更好或不变**，这是【无混杂】的单变量形式。

改动：+1 个可学习恒等细化器 U（约 7K 参数）。其余（refiner 形状 16ch/2bl、数据、目标、w_gan、unsharp）逐项与 V4 相同。

预登记判据（跑完不改）：
  全链 clipiqa > 0.8118  => 净胜当前最优 V4t1
  == 0.8118 附近          => U 没起作用（恒等就是最优）=> 轴 CLOSED
  < 0.8042               => 明确负
用法: python s1b_train.py [--tag s1b] [--steps 14000] [--init_up ""]
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="s1b")
ap.add_argument("--steps", type=int, default=14000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--w_gan", type=float, default=0.1)
ap.add_argument("--ch", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--init_ref", default=r"D:\models\refine_v1\refiner_thin2_s6000.pt",
                help="refiner 初始化（默认 = 已提交 V4 的 refiner => 与 U 恒等一起精确复现 V4）")
ap.add_argument("--init_up", default="", help="从上一阶段 (U,G) 继续")
a = ap.parse_args()

dev = "cuda"
IN = Path(r"D:\models\refine_v1\in")
TGT = Path(r"D:\datasets\distill256\tgt")
OUT = Path(r"D:\models\refine_v1")
LOG = OUT / ("train_%s.log" % a.tag)
torch.manual_seed(0); np.random.seed(0)


class UIdent(nn.Module):
    """可学习恒等细化器，输入输出同域（S 域）。zero-init => 起点精确为恒等。"""

    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 3, 3, 1, 1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


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
        mult = 1; prev = ch
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


U = UIdent(32).to(dev)
G = Refiner(a.ch, a.blocks).to(dev)
if a.init_up and Path(a.init_up).exists():
    sd = torch.load(a.init_up, map_location="cpu", weights_only=False)
    U.load_state_dict(sd["U"]); G.load_state_dict(sd["net"])
    log("[init] (U,G) <- %s steps=%s" % (Path(a.init_up).name, sd.get("steps")))
elif a.init_ref and Path(a.init_ref).exists():
    sd = torch.load(a.init_ref, map_location="cpu", weights_only=False)
    G.load_state_dict(sd["net"])
    log("[init] G <- %s (steps=%s) ; U = 恒等(zero-init) => 起点精确等于已提交 V4"
        % (Path(a.init_ref).name, sd.get("steps")))
D = PatchD().to(dev)
log("[model] U=%d params G=%d params D=%d params" % (
    sum(p.numel() for p in U.parameters()), sum(p.numel() for p in G.parameters()),
    sum(p.numel() for p in D.parameters())))

import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dev).eval()
for p in lp.parameters():
    p.requires_grad_(False)
gp = list(U.parameters()) + list(G.parameters())
optG = torch.optim.AdamW(gp, lr=a.lr, betas=(0.9, 0.99))
optD = torch.optim.AdamW(D.parameters(), lr=a.lr_d, betas=(0.9, 0.99))
schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=a.steps, eta_min=a.lr * 0.05)
schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=a.steps, eta_min=a.lr_d * 0.05)
scG = torch.amp.GradScaler("cuda"); scD = torch.amp.GradScaler("cuda")
files = sorted(IN.glob("*.jpg"))
log("[data] %d 对（随机 %d 裁剪，S 域）" % (len(files), a.crop))


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
            fake = G(U(xin))
        pr = D(tgt); pf = D(fake)
        lossD = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
    optD.zero_grad(set_to_none=True); scD.scale(lossD).backward(); scD.step(optD); scD.update()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(U(xin))
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
        torch.save({"U": U.state_dict(), "net": G.state_dict(), "ch": a.ch, "blocks": a.blocks,
                    "steps": step, "kind": "uident"}, OUT / ("uident_%s_s%d.pt" % (a.tag, step)))
        log("[save] uident_%s_s%d.pt" % (a.tag, step))
log("S1B_TRAIN_DONE")
