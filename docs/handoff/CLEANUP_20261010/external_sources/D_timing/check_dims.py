from pathlib import Path
from PIL import Image
import numpy as np
V = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
T = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
print('--- val ---')
for p in sorted(V.glob('*')):
    im = Image.open(p)
    print(f'{p.name:20s} {im.size} {im.mode} {p.stat().st_size/1e6:.2f}MB')
print('--- test (first 10) ---')
for p in sorted(T.glob('*'))[:10]:
    im = Image.open(p)
    print(f'{p.name:20s} {im.size} {im.mode} {p.stat().st_size/1e6:.2f}MB')
print('test count', len(list(T.glob('*'))))
