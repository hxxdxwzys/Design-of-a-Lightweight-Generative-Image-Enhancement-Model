# -*- coding: utf-8 -*-
"""多 ckpt 输出集成评估: 平均多个 ckpt 的验证集渲染, 再算 lpips+clipiqa+公式质量分
用法: python eval_ensemble.py <dir1> <dir2> [dir3 ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:\tmp_timing')

VAL = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集')


def load(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    assert len(dirs) >= 2
    import pyiqa
    lp = pyiqa.create_metric('lpips', device='cuda')
    cq = pyiqa.create_metric('clipiqa', device='cuda')
    for case in (1, 2, 3):
        gt_p = VAL / f'case{case}_gt.png'
        if not gt_p.exists():
            gt_p = VAL / f'case{case}_gt.jpg'
        gt = load(gt_p).clamp(1e-4, 1 - 1e-4).cuda()
        acc = None
        for d in dirs:
            pred = load(d / f'case{case}_pred.jpg').clamp(1e-4, 1 - 1e-4).cuda()
            acc = pred if acc is None else acc + pred
        ens = (acc / len(dirs)).clamp(1e-4, 1 - 1e-4)
        l = float(lp(ens, gt).mean().item())
        c = float(cq(ens).mean().item())
        print(f'case{case}: lpips={l:.4f} clipiqa={c:.4f}')
        # 保存集成输出
        arr = (ens[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(rf'D:\tmp_timing\ens_{len(dirs)}mix\case{case}_pred.jpg', quality=98)


if __name__ == '__main__':
    main()
