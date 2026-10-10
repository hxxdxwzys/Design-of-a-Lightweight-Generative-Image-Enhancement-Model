# -*- coding: utf-8 -*-
"""逐层对比特征: repo extract_layers vs runner._features"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import torch
MD = r'D:/models/pkg_PixB/model_dir'
sys.path.insert(0, MD); sys.path.insert(0, r'D:/models/newsrc/PixRestore_code')
import importlib.util
spec = importlib.util.spec_from_file_location('pixrunner', os.path.join(MD, 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.Runner(MD)
from pixrestore.vision import extract_layers, preprocess_dinov2
x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1).half()
with torch.no_grad():
    ref = extract_layers(R.venc, x, R.layers, R.enc_size)
    my = R._features(x)
    print('layers:', R.layers, 'n_ref', len(ref), 'n_my', len(my), flush=True)
    for i, (a, b) in enumerate(zip(ref, my)):
        print(f'  layer idx {i}: ref {tuple(a.shape)} {a.dtype} mean {a.float().mean():.4f} | '
              f'my {tuple(b.shape)} {b.dtype} mean {b.float().mean():.4f} | maxdiff {float((a.float()-b.float()).abs().max()):.4f}', flush=True)
    # 直接看 normed 输入是否一致
    xa = preprocess_dinov2(x, R.enc_size)
    raw = torch.nn.functional.interpolate(x.float().add(1).mul(0.5), size=(R.enc_size, R.enc_size),
                                          mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
    xb = ((raw - mod._MEAN.float()) / mod._STD.float()).half()
    print('input maxdiff:', float((xa.half().float() - xb.float()).abs().max()), flush=True)
print('DIAG2_DONE')
