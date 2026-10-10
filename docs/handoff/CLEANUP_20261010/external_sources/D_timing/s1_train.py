# -*- coding: utf-8 -*-
"""S1（最后一次冲分）：在【双三次 256->512 上采样】与 refiner 之间插入【可学习上采样器】，联合训练。

假设：整条链上唯一无参数、无学习的环节是 256->512 的双三次上采样；它造成的模糊是 refiner 的瓶颈。
      给 refiner 一个可学习的上采样前端（U），它就能把 11K 参数的能力用于"合成细节"而非"去模糊"。
      证据：refiner 增益随底图清晰度单调下降（0.63->+0.174、0.67->+0.130、0.70->+0.08），
            而三档容量实验都跑在同等模糊的底图上 => 只扫了容量，没扫锐度。

改动（单变量）：只新增 U（3->12ch，PixelShuffle 2x，残差连到 bicubic，zero-init 输出 => 起点≈现役）。
              学生冻结；数据/目标/refiner 形状/步数/w_gan 与已提交 V4 的 refiner 配方逐项相同。

预登记判据（跑完不改）：
  全链 clipiqa >= 0.83      => 活
  > 0.8118                  => 净胜当前最优（V4t1）
  <= 0.8042                 => CLOSED
用法: python s1_train.py [--tag up1] [--steps 14000] [--ch 16] [--blocks 2] [--lr 1e-4]
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="up1")
ap.add_argument("--steps", type=int, default=14000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--w_lpips", type=float, default=0.2)
ap.add_argument("--w_gan", type=float, default=0.1)
ap.add_argument("--ch", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--size", type=int, default=512, help="学生输出域尺寸(=refiner 工作域)")
ap.add_argument("--init_up", default="", help="从上一阶段的 (U,G) 检查点继续（两阶段训练用）")
a = ap.parse_args()

dev = "cuda"
IN = Path(r"D:\models\refine_v1\in")            # 学生 512 域输出（与已提交 V4 的 refiner 同源）
TGT = Path(r"D:\datasets\distill256\tgt")       # M24 目标
OUT = Path(r"D:\models\refine_v1")
LOG = OUT / ("train_%s.log" % a.tag)
STU = r"D:\models\tiny_v1\v21_s21000.pt"
torch.manual_seed(0); np.random.seed(0)
from diffusers import UNet2DModel


class Up2(nn.Module):
    """可学习 2x 上采样：残差连到 bicubic（zero-init 输出 => 训练起点等价于现役 bicubic）。"""
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 12, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        return (base + self.ps(self.c2(self.act(self.c1(x))))).clamp(-1, 1)


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
        for _ in range(n_layers):
            mult = min(2 ** _, 8)
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


def down(x):
    """复刻推理链的 512 -> 256 双三次下采样。"""
    return F.interpolate(x, size=(x.shape[-1] // 2, x.shape[-2] // 2), mode="bicubic", align_corners=False)


# 冻结的学生（只用来产 S 域底图；S 由 --size 指定）
d = torch.load(STU, map_location="cpu", weights_only=False)
ch_s = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch_s, down_block_types=("DownBlock2D",) * len(ch_s),
                   up_block_types=("UpBlock2D",) * len(ch_s))
unet.load_state_dict(d["unet"]); unet = unet.to(dev).eval().half()
for p in unet.parameters():
    p.requires_grad_(False)
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5

U = Up2(16).to(dev)
G = Refiner(a.ch, a.blocks).to(dev)
if a.init_up and Path(a.init_up).exists():
    sd = torch.load(a.init_up, map_location="cpu", weights_only=False)
    U.load_state_dict(sd["U"]); G.load_state_dict(sd["net"])
    print("[init] (U,G) <- %s steps=%s" % (Path(a.init_up).name, sd.get("steps")), flush=True)
D = PatchD().to(dev)
log("[model] U=%d params  G=%d params  D=%d params  (student frozen)"
    % (sum(p.numel() for p in U.parameters()), sum(p.numel() for p in G.parameters()),
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
log("[data] %d 对（随机 %d 裁剪）" % (len(files), a.crop))


def to_t(arr):
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)


def sample():
    ins, tg = [], []
    for _ in range(a.bs):
        f = files[np.random.randint(len(files))]
        i = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        g = np.asarray(Image.open(TGT / (f.stem + ".jpg")).convert("RGB"), dtype=np.uint8)
        c = a.crop
        y0 = np.random.randint(0, i.shape[0] - c + 1); x0 = np.random.randint(0, i.shape[1] - c + 1)
        ins.append(i[y0:y0 + c, x0:x0 + c]); tg.append(g[y0:y0 + c, x0:x0 + c])
    return to_t(ins), to_t(tg)


t0 = time.time(); rg = rd = 0.0
for step in range(1, a.steps + 1):
    xin512, tgt = sample()                      # 512 域（crop=512）：in 本身即学生的 512 域输出
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            fake = G(U(down(xin512)))           # 复刻推理链：512 ->(双三次↓256)-> U↑512 -> refiner@512
        pr = D(tgt); pf = D(fake)
        lossD = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
    optD.zero_grad(set_to_none=True); scD.scale(lossD).backward(); scD.step(optD); scD.update()
    with torch.autocast("cuda", dtype=torch.float16):
        out = G(U(down(xin512)))
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
        torch.save({"U": U.state_dict(), "net": G.state_dict(), "ch": a.ch, "blocks": a.blocks, "steps": step},
                   OUT / ("upref_%s_s%d.pt" % (a.tag, step)))
        log("[save] upref_%s_s%d.pt" % (a.tag, step))
log("S1_TRAIN_DONE")
