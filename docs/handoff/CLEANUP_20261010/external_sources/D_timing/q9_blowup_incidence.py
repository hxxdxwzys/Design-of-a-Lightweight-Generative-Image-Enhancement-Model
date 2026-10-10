# -*- coding: utf-8 -*-
"""q9_blowup_incidence.py — 统计各 runner 在 512 输入上的"值域越界"发生率 (两种输入口径)."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
dev = 'cuda'
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:20]]
PKGS = [('v4g', r'D:/models/pkg_TinyV4g', 128), ('v4g256', r'D:/models/pkg_TinyV4g', 256),
        ('v7gan128', r'D:/models/pkg_TinyV7gan', 128), ('v7gan256', r'D:/models/pkg_TinyV7gan', 256),
        ('v11b128', r'D:/models/pkg_TinyV11b', 128), ('v5d128', r'D:/models/pkg_TinyV5d', 128),
        ('v9b512', r'D:/models/pkg_TinyV9b', 512)]
mods = {}
for tag, pkg, n in PKGS:
    if pkg not in mods:
        mods[pkg] = torch.jit.load(str(Path(pkg)/'model_dir'/'your_model.pt'), map_location=dev).eval()
for tag, pkg, n in PKGS:
    m = mods[pkg]
    stats = {'crop': [0, 0, 0.0], 'rsz': [0, 0, 0.0]}   # [图片数带越界, 越界像素总数, 最大|值|]
    for mode in ('crop', 'rsz'):
        for nm in names:
            im = Image.open(TEST/(nm+'.jpg')).convert('RGB')
            W, H = im.size
            if mode == 'crop':
                im2 = im.crop((W//2-256, H//2-256, W//2+256, H//2+256)).resize((512, 512), Image.BICUBIC)
            else:
                im2 = im.resize((512, 512), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im2).copy()).permute(2,0,1).unsqueeze(0).cuda().half()/127.5-1
            with torch.no_grad():
                xs = F.interpolate(x, size=(n,n), mode='bicubic', align_corners=False)
                y = m(xs)
                if y.shape[-2:] != x.shape[-2:]:
                    y = F.interpolate(y, size=x.shape[-2:], mode='bicubic', align_corners=False)
            f = y.float()
            bad = int((f.abs() > 1.15).sum().item())
            mx = float(f.abs().max().item())
            s = stats[mode]
            if bad > 0: s[0] += 1
            s[1] += bad; s[2] = max(s[2], mx)
            del x, xs, y, f
    print('%-10s crop口径: %2d/20 图有越界, 共 %6d 像素, max|v|=%.2f | resize口径: %2d/20, 共 %6d, max=%.2f' % (
        tag, stats['crop'][0], stats['crop'][1], stats['crop'][2], stats['rsz'][0], stats['rsz'][1], stats['rsz'][2]), flush=True)
print('Q9_DONE')
