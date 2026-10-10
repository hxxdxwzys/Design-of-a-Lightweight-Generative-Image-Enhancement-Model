# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
iq = {k: pyiqa.create_metric(k, device='cuda') for k in ('maniqa', 'clipiqa')}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:30]]
for tag, d, pat in [('probeA_test_sc0.4', r'D:/models/probe_a/pred_sc0.4', 'case{n}.png'),
                    ('M24_test_sc0.8', r'D:/models/final_compliant/output_dir', 'case{n}.jpg')]:
    acc = {k: [] for k in iq}; lv = []
    for nm in names:
        idx = ''.join(c for c in nm if c.isdigit())
        p = Path(d) / pat.format(n=idx)
        if not p.exists(): continue
        im = Image.open(p).convert('RGB')
        t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)
        for k, m in iq.items():
            with torch.no_grad(): acc[k].append(float(m(t).item()))
        g = np.asarray(im.convert('L'), dtype=np.float64)
        lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
        lv.append(float(lap.var()))
        del t
    print('%-20s n=%d maniqa=%.4f clipiqa=%.4f lapvar=%.1f' % (
        tag, len(lv), float(np.mean(acc['maniqa'])), float(np.mean(acc['clipiqa'])), float(np.mean(lv))), flush=True)
print('PROBEA_DONE')
