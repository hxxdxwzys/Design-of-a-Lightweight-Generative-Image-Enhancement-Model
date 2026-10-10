# -*- coding: utf-8 -*-
"""锐度/细节能量 对照 (2026-09-13): 官方分 vs 高频细节量
假设 H_sharp: 官方质量项 ≈ 对细节/锐度敏感、且与 NIQE 反向。
做法: 对 5 个已交包的 4K 输出 (100 张) 抽 20 张, 算 LQ 相同位置的细节增益。
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = {
 'M24     (43.6951)': Path(r'D:/models/final_compliant/output_dir'),
 'probeA  (43.2837)': Path(r'D:/models/final_compliant_probeA_sc04/output_dir'),
 'sc00    (37.6316)': Path(r'D:/models/final_compliant_M24sc00/output_dir'),
 'avg2    (36.6414)': Path(r'D:/models/final_compliant_M24avg2/output_dir'),
 'p4      ( 7.1829)': Path(r'D:/models/final_compliant_M24p4/output_dir'),
 '1.4B m0s7(30.9363)': Path(r'D:/models/final_compliant_m0s7/output_dir'),
 'LQ(输入, 参照)': TEST,
}
IDX = list(range(1, 101, 5))  # 20 张, 等间隔

def feats(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
    lap = ndimage.laplace(g)
    lv = float(lap.var())
    gy, gx = np.gradient(g)
    gm = float(np.mean(np.hypot(gx, gy)))
    lo = ndimage.gaussian_filter(g, 2.0)
    hf = float(np.sqrt(np.mean((g - lo) ** 2)))
    return lv, gm, hf

out = {}
for k, d in PKGS.items():
    acc = []
    for i in IDX:
        p = d / f'case{i}.jpg'
        if not p.exists():
            p = d / f'case{i}.png'
        if not p.exists():
            continue
        acc.append(feats(p))
    a = np.array(acc)
    out[k] = {'n': len(a), 'lapvar': a[:,0].mean(), 'gradmean': a[:,1].mean(), 'hf': a[:,2].mean()}
    print(k, out[k], flush=True)

# 细节增益 = 输出HF / LQ同图HF
lq = out['LQ(输入, 参照)']
print('\n=== 细节增益 (相对 LQ, 同 20 张) ===')
for k, v in out.items():
    if k.startswith('LQ'):
        continue
    print(f'{k}  lapvar x{v["lapvar"]/lq["lapvar"]:.2f}  grad x{v["gradmean"]/lq["gradmean"]:.2f}  hf x{v["hf"]/lq["hf"]:.2f}')

Path(r'D:/tmp_timing/hf_sharpness.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('DONE')
