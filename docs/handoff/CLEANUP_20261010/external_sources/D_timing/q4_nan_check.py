# -*- coding: utf-8 -*-
"""检查各 JIT runner 在 512 输入上的 NaN/Inf 与内部数值范围."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
dev='cuda'
im = Image.open(TEST/'case12.jpg').convert('RGB')
W,H = im.size
im = im.crop((W//2-256, H//2-256, W//2+256, H//2+256))
a = np.asarray(im).astype(np.float32)/255.0
x = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).cuda().half()*2-1
PKGS = [('v4g', r'D:/models/pkg_TinyV4g', 128), ('v5d', r'D:/models/pkg_TinyV5d', 256),
        ('v7gan', r'D:/models/pkg_TinyV7gan', 256), ('v9b', r'D:/models/pkg_TinyV9b', 512),
        ('v11b', r'D:/models/pkg_TinyV11b', 256), ('v1', r'D:/models/pkg_TinyV1', 512)]
for tag, pkg, n in PKGS:
    try:
        m = torch.jit.load(str(Path(pkg)/'model_dir'/'your_model.pt'), map_location=dev).eval()
        sd = m.state_dict()
        bufs = {k: float(v) for k, v in sd.items() if k in ('sq_a','sq_1a','tt') or v.numel()==1}
        with torch.no_grad():
            xs = F.interpolate(x, size=(n,n), mode='bicubic', align_corners=False)
            y = m(xs)
            if y.shape[-2:] != x.shape[-2:]:
                y = F.interpolate(y, size=x.shape[-2:], mode='bicubic', align_corners=False)
        f = y.float()
        nan = int(torch.isnan(f).sum().item()); inf = int(torch.isinf(f).sum().item())
        print('%-6s NATIVE=%3d  bufs=%s  out[min=%.3f max=%.3f]  NaN=%d Inf=%d  (总元素 %d)' % (
            tag, n, {k: round(v,6) for k,v in bufs.items()}, float(f.min()), float(f.max()), nan, inf, f.numel()))
        if nan or inf:
            mask = torch.isnan(f[0,0]) | torch.isinf(f[0,0])
            ys, xs_ = torch.nonzero(mask, as_tuple=True)
            if len(ys): print('        坏点范围: rows %d-%d cols %d-%d, 数量 %d' % (ys.min(), ys.max(), xs_.min(), xs_.max(), len(ys)))
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(tag, 'ERR', repr(e)[:200])
print('NAN_CHECK_DONE')
