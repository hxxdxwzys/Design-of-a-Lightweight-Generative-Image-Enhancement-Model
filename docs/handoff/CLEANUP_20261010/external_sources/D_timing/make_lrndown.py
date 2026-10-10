# -*- coding: utf-8 -*-
"""生成 train_tiny_lrndown.py（修正版）：512->256 固定双三次 -> 可学习抗锯齿下采样。
单变量；`--lrn_down 0` 时与原件逐行等价。
"""
from pathlib import Path

SRC = Path(r"D:\tmp_timing\train_tiny_gan.py")
DST = Path(r"D:\tmp_timing\train_tiny_lrndown.py")
t = SRC.read_text(encoding="utf-8-sig")


def sub(old, new, tag):
    global t
    assert old in t, "anchor missing: " + tag
    t = t.replace(old, new, 1)


# 1) 参数（插在 --d_band 的整块注释之后，避免切断注释）
sub("""    ap.add_argument('--d_band', type=int, default=0)     # 1 => 判别器输入先做 band-limit(avgpool2+nearest 上采样),
                                                         #      去掉"清晰度"这个平凡线索, 让 D 只判结构/伪影 (修 D 秒杀退化)""",
    """    ap.add_argument('--d_band', type=int, default=0)     # 1 => 判别器输入先做 band-limit(avgpool2+nearest 上采样),
                                                         #      去掉"清晰度"这个平凡线索, 让 D 只判结构/伪影 (修 D 秒杀退化)
    ap.add_argument('--lrn_down', type=int, default=0,
                    help='1=用可学习抗锯齿下采样替换固定双三次 512->256')""", "arg")

# 2) 模块
sub("class PatchD(nn.Module):",
    '''class LearnDown(nn.Module):
    """可学习抗锯齿下采样：5x5 深度可分离高斯核 -> 步长 2。
    替代固定 F.interpolate(bicubic)，让网络自己决定「该怎么丢信息」。"""

    def __init__(self, ch=3):
        super().__init__()
        ar = torch.arange(5, dtype=torch.float32) - 2.0
        k = torch.exp(-(ar * ar) / 2.0)
        k = (k / k.sum()).view(1, 1, 1, 5).repeat(ch, 1, 1, 1)
        self.k = nn.Parameter(k.clone())          # 初始化为高斯，可学

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        c = x.shape[1]
        w = self.k[:c].float()
        w = w / w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        y = F.pad(xf, (2, 2, 0, 0), mode="reflect")
        y = F.conv2d(y, w, groups=c)
        y = F.pad(y, (0, 0, 2, 2), mode="reflect")
        y = F.conv2d(y, w.transpose(2, 3).contiguous(), groups=c)
        return y[:, :, ::2, ::2].to(dt)


class PatchD(nn.Module):''', "module")

# 3) 下采样器实例化（放在 dev 定义之后的模块级，D 定义之前）
sub("    D = PatchD().to(dev)",
    "    DN = LearnDown().to(dev) if a.lrn_down else None\n"
    "    if DN is not None:\n"
    "        print('[lrn_down] learnable anti-aliased downsampler ON (%d params)'\n"
    "              % sum(p.numel() for p in DN.parameters()), flush=True)\n"
    "    D = PatchD().to(dev)", "instantiate")

# 4) 优化器包含 DN
sub("    optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99))",
    "    _gparams = list(G.parameters()) + (list(DN.parameters()) if DN is not None else [])\n"
    "    optG = torch.optim.AdamW(_gparams, lr=a.lr, betas=(0.9, 0.99))", "optim")

# 5) 下采样调用点（DN 内部自己做 dtype 归一，故这里直接传 tensor）
sub("""        if hq.shape[-1] != a.res:   # 降到学生实际运行的域
            hq = F.interpolate(hq, size=(a.res, a.res), mode='bicubic', align_corners=False)
            lq = F.interpolate(lq, size=(a.res, a.res), mode='bicubic', align_corners=False)""",
    """        if hq.shape[-1] != a.res:   # 降到学生实际运行的域
            if DN is not None:
                hq = DN(hq)                     # 可学习抗锯齿下采样（内部做 dtype 归一）
                lq = DN(lq)
            else:
                hq = F.interpolate(hq, size=(a.res, a.res), mode='bicubic', align_corners=False)
                lq = F.interpolate(lq, size=(a.res, a.res), mode='bicubic', align_corners=False)""", "call")

# 6) 存盘带 DN
sub("""            torch.save({'unet': G.state_dict(), 'ch': ch, 'res': a.res, 'alpha': float(alpha),
                        'steps': step, 'gan': True}, OUT / f'{a.tag}_s{step}.pt')""",
    """            _ck = {'unet': G.state_dict(), 'ch': ch, 'res': a.res, 'alpha': float(alpha),
                   'steps': step, 'gan': True, 'lrn_down': a.lrn_down}
            if DN is not None:
                _ck['down'] = DN.state_dict()
            torch.save(_ck, OUT / f'{a.tag}_s{step}.pt')""", "save")

DST.write_text(t, encoding="utf-8")
print("written:", DST)
for i, line in enumerate(t.splitlines(), 1):
    if "lrn_down" in line or "DN" in line or "LearnDown" in line or "_gparams" in line:
        print("  %4d | %s" % (i, line.rstrip()))
