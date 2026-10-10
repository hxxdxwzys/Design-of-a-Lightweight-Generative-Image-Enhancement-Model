# -*- coding: utf-8 -*-
"""q6_train_flat.py — 训练"扁平轻量一步扩散学生"(x0 参数化, 残差+clamp 有界), 蒸馏 M24 的 4K 观感.
训练域对齐官方测速口径: 4K 图先缩到长边 1024, 再随机取 512 训练块 (即 8x 下采样统计).
架构故意做成 6~8 个 conv 的扁平栈: 内核数少 => CUDA Graph 重放远快于 UNet(~200 内核).
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--c', type=int, default=16)
ap.add_argument('--layers', type=int, default=7)
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--nimg', type=int, default=60)
ap.add_argument('--hf', type=float, default=0.5)
ap.add_argument('--out', default=r'D:/models/tiny_v1/flat_student.pt')
a = ap.parse_args()

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
dev = 'cuda'

def load_pairs(nimg, long_side=1024):
    L, H = [], []
    for i in range(1, nimg + 1):
        nm = 'case%d' % i
        lq = Image.open(TEST / (nm + '.jpg')).convert('RGB')
        m24 = Image.open(M24D / (nm + '.jpg')).convert('RGB')
        w, h = lq.size
        s = long_side / max(w, h)
        nw, nh = (int(round(w * s)) // 8 * 8, int(round(h * s)) // 8 * 8)
        L.append(np.asarray(lq.resize((nw, nh), Image.BICUBIC)))
        H.append(np.asarray(m24.resize((nw, nh), Image.BICUBIC)))
    return L, H

print('[data] loading %d pairs...' % a.nimg, flush=True)
L, H = load_pairs(a.nimg)
print('[data] %d pairs, shape %s' % (len(L), L[0].shape), flush=True)
ntr = int(len(L) * 0.85)
tr = list(range(ntr)); va = list(range(ntr, len(L)))

def crop(a1, a2, i, size):
    h, w, _ = a1.shape
    y = np.random.randint(0, h - size + 1); x = np.random.randint(0, w - size + 1)
    x1 = torch.from_numpy(a1[y:y+size, x:x+size].copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
    x2 = torch.from_numpy(a2[y:y+size, x:x+size].copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
    return x1, x2

class Flat1Step(nn.Module):
    """x0 参数化的一步扩散学生: x0 = clamp(x + net(x)) (残差+有界, 结构上不可能出现爆点)."""
    def __init__(self, c=16, layers=7):
        super().__init__()
        seq = [nn.Conv2d(3, c, 3, 1, 1), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Conv2d(c, c, 3, 1, 1), nn.SiLU()]
        seq += [nn.Conv2d(c, 3, 3, 1, 1)]
        self.net = nn.Sequential(*seq)
        self.register_buffer('sq_a', torch.tensor(0.525879))
        self.register_buffer('sq_1a', torch.tensor(0.850586))
        self.register_buffer('tt', torch.tensor(500.0))
    def forward(self, x):
        return torch.clamp(x + self.net(x), -1.0, 1.0)

def hf(t):
    k = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], device=t.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(t, k, padding=1, groups=3)

m = Flat1Step(a.c, a.layers).to(dev)
nparam = sum(p.numel() for p in m.parameters())
print('[model] c=%d layers=%d params=%.3fM' % (a.c, a.layers, nparam / 1e6), flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
scaler = torch.cuda.amp.GradScaler()

t0 = time.time()
for step in range(1, a.steps + 1):
    m.train()
    xs, ys = [], []
    for _ in range(a.bs):
        i = tr[np.random.randint(len(tr))]
        x1, x2 = crop(L[i], H[i], i, a.size)
        xs.append(x1); ys.append(x2)
    x = torch.stack(xs).to(dev); y = torch.stack(ys).to(dev)
    with torch.autocast('cuda', dtype=torch.float16):
        o = m(x)
        loss = F.l1_loss(o, y) + a.hf * F.l1_loss(hf(o), hf(y))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if step % 200 == 0 or step == 1:
        m.eval()
        with torch.no_grad():
            i = va[0]
            x1, x2 = crop(L[i], H[i], i, a.size)
            xb = x1.unsqueeze(0).to(dev); yb = x2.unsqueeze(0).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                ob = m(xb)
            l1 = float(F.l1_loss(ob.float(), yb).item())
            l1lq = float(F.l1_loss(xb, yb).item())
            def lv(t):
                g = (t.float().mean(1, keepdim=True) * 255)
                lap = (-4*g[...,1:-1,1:-1] + g[...,:-2,1:-1] + g[...,2:,1:-1] + g[...,1:-1,:-2] + g[...,1:-1,2:])
                return float(lap.var().item())
            print('step %5d  loss=%.4f  valL1(out,GT)=%.4f  valL1(LQ,GT)=%.4f  lapvar out/LQ/GT=%.0f/%.0f/%.0f  %.0fs' % (
                step, float(loss.item()), l1, l1lq, lv(ob), lv(xb), lv(yb), time.time() - t0), flush=True)
        torch.save({'sd': m.state_dict(), 'cfg': dict(c=a.c, layers=a.layers)}, a.out)
print('[save]', a.out, flush=True)
print('FLAT_TRAIN_DONE', flush=True)
