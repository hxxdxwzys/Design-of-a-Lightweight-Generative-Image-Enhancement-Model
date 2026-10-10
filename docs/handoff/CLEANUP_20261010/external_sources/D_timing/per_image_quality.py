# -*- coding: utf-8 -*-
"""100 图级逐图质量分析:
1. s6k 输出的 val 逐 case lpips/clipiqa (对照 HYPIR)
2. 100 测试图逐图 clipiqa: LQ vs s6k vs HYPIR-q98
输出: per_image_report.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VAL_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集')
TEST_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
S6K_VAL = Path(r'D:\tmp_timing\cnn_val_s6000')
S6K_OUT = Path(r'D:\models\cnn_out_s6k')
HYPIR_OUT = Path(r'D:\models\hypir_q98_out\result')


def load(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    import pyiqa
    device = 'cuda'
    lp = pyiqa.create_metric('lpips', device=device)
    cq = pyiqa.create_metric('clipiqa', device=device)

    # 1) s6k val 逐 case
    print('== s6k val per-case ==', flush=True)
    val = {}
    for case in (1, 2, 3):
        gt_p = VAL_DIR / f'case{case}_gt.png'
        if not gt_p.exists():
            gt_p = VAL_DIR / f'case{case}_gt.jpg'
        pred = load(S6K_VAL / f'case{case}_pred.jpg').clamp(1e-4, 1 - 1e-4).cuda()
        gt = load(gt_p).clamp(1e-4, 1 - 1e-4).cuda()
        lp_v = float(lp(pred, gt).mean().item())
        cq_v = float(cq(pred).mean().item())
        val[case] = {'lpips': lp_v, 'clipiqa': cq_v}
        print(f'  case{case}: lpips={lp_v:.4f} clipiqa={cq_v:.4f}', flush=True)

    # 2) 100 测试图逐图 clipiqa
    print('== test per-image clipiqa ==', flush=True)
    rep = {}
    t0 = time.time()
    done = 0
    for i, p in enumerate(sorted(TEST_DIR.glob('*.jpg'))):
        name = p.stem
        lq = load(p).clamp(1e-4, 1 - 1e-4).cuda()
        s6k = load(S6K_OUT / f'{name}.jpg').clamp(1e-4, 1 - 1e-4).cuda()
        hy = load(HYPIR_OUT / f'{name}.png').clamp(1e-4, 1 - 1e-4).cuda()
        rep[name] = {
            'lq': float(cq(lq).mean().item()),
            's6k': float(cq(s6k).mean().item()),
            'hypir': float(cq(hy).mean().item()),
        }
        done += 1
        if done % 10 == 0:
            with open(r'D:\tmp_timing\per_image_report_partial.json', 'w') as f:
                json.dump({'val_s6k': val, 'test_clipiqa': rep}, f)
            print(f'  {done}/100 ({time.time()-t0:.0f}s, {((time.time()-t0)/done):.1f}s/img)', flush=True)
    print('== done ==', flush=True)
    with open(r'D:\tmp_timing\per_image_report.json', 'w') as f:
        json.dump({'val_s6k': val, 'test_clipiqa': rep}, f)
    print('saved per_image_report.json', flush=True)


if __name__ == '__main__':
    main()
