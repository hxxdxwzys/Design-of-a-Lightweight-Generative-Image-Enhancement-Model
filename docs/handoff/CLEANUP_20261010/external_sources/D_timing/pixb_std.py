# -*- coding: utf-8 -*-
"""核实: PixB runner 在【真实图像】与【随机张量】上的输出统计"""
import os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch
from PIL import Image
MD = r'D:/models/pkg_PixB/model_dir'
sys.path.insert(0, MD)
import importlib.util
spec = importlib.util.spec_from_file_location('pixrunner', os.path.join(MD, 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.Runner(MD)
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
for tag, arr in [('真实图像(case13 crop)', np.asarray(Image.open(TEST/'case13.jpg').convert('RGB'), dtype=np.uint8)[1000:1512, 1000:1512])]:
    t = torch.from_numpy(arr.copy()).permute(2,0,1)[None].float().div_(127.5).sub_(1)
    y = R.infer(t)
    print(f'{tag}: out std={y.std().item():.3f} mean={y.mean().item():.3f} '
          f'min={y.min().item():.3f} max={y.max().item():.3f} shape={tuple(y.shape)}', flush=True)
x = torch.rand(1,3,512,512)*2-1
y = R.infer(x)
print(f'随机张量: out std={y.std().item():.3f} mean={y.mean().item():.3f}', flush=True)
print('REF_STD_DONE')
