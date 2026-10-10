# -*- coding: utf-8 -*-
"""生成 train_refiner_msd.py：在原 train_refiner_gan2.py 上加 --msd（多尺度判别器）。
只改判别器形状与损失聚合，其余逐行不动。"""
import io
from pathlib import Path

SRC = Path(r"D:\tmp_timing\train_refiner_gan2.py")
DST = Path(r"D:\tmp_timing\train_refiner_msd.py")
t = SRC.read_text(encoding="utf-8-sig")   # 源文件可能带 BOM（之前用 PowerShell Set-Content 写过）

# 1) 新增 --msd
anchor1 = 'ap.add_argument("--in_dir"'
assert anchor1 in t, "anchor1 missing"
t = t.replace(anchor1,
              'ap.add_argument("--msd", type=int, default=0,\n'
              '                help="1=多尺度判别器(512+256+128)，损失除以尺度数以保持 gan 量级不变")\n'
              + anchor1, 1)

# 2) 判别器构造
anchor2 = "D = PatchD().to(dev)"
assert anchor2 in t, "anchor2 missing"
msd_cls = '''class MSD(nn.Module):
    """多尺度 PatchGAN：512/256/128 三个尺度各自判别；损失取平均 => 量级与单尺度一致，
    因此【只改监督的分辨率覆盖】，不改对抗强度。"""

    def __init__(self):
        super().__init__()
        self.d1 = PatchD(); self.d2 = PatchD(); self.d3 = PatchD()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        o1 = self.d1(x)
        x2 = self.pool(x)
        o2 = self.d2(x2)
        x3 = self.pool(x2)
        o3 = self.d3(x3)
        return o1, o2, o3


if a.msd:
    D = MSD().to(dev)
    print("[msd] multi-scale discriminator ON (512/256/128); loss averaged over scales", flush=True)
else:
    D = PatchD().to(dev)'''
t = t.replace(anchor2, msd_cls, 1)

# 3) D 损失
anchor3 = """        pred_real = D(tgt); pred_fake = D(fake)
        lossD = 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                       F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))"""
assert anchor3 in t, "anchor3 missing"
new3 = """        pr = D(tgt); pfk = D(fake)
        if isinstance(pr, tuple):
            lossD = sum(0.5 * (F.mse_loss(u, torch.ones_like(u)) + F.mse_loss(v, torch.zeros_like(v)))
                        for u, v in zip(pr, pfk)) / len(pr)
        else:
            lossD = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) +
                           F.mse_loss(pfk, torch.zeros_like(pfk)))"""
t = t.replace(anchor3, new3, 1)

# 4) G 的对抗损失
anchor4 = """        pf = D(out)
        gan = F.mse_loss(pf, torch.ones_like(pf))"""
assert anchor4 in t, "anchor4 missing"
new4 = """        pfo = D(out)
        if isinstance(pfo, tuple):
            gan = sum(F.mse_loss(o, torch.ones_like(o)) for o in pfo) / len(pfo)
        else:
            gan = F.mse_loss(pfo, torch.ones_like(pfo))"""
t = t.replace(anchor4, new4, 1)

DST.write_text(t, encoding="utf-8")
print("written:", DST)
for i, line in enumerate(t.splitlines(), 1):
    if "msd" in line.lower() or "MSD" in line or "isinstance" in line:
        print("  %4d | %s" % (i, line))
