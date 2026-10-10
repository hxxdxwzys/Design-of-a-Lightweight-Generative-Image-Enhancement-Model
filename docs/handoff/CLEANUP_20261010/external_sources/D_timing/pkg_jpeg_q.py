# -*- coding: utf-8 -*-
"""各包输出 jpg 的实际质量因子 (量化表反推) —— 排除"HF差异只是保存质量差异"的混淆"""
from pathlib import Path
from PIL import Image
import io

DIRS = {
 'M24': r'D:/models/final_compliant/output_dir',
 'probeA': r'D:/models/final_compliant_probeA_sc04/output_dir',
 'sc00': r'D:/models/final_compliant_M24sc00/output_dir',
 'avg2': r'D:/models/final_compliant_M24avg2/output_dir',
 'p4': r'D:/models/final_compliant_M24p4/output_dir',
 'm0s1(14b)': r'D:/models/final_compliant_14b/output_dir',
 'g3w18s1': r'D:/models/final_compliant_g3w18s1/output_dir',
 'm0s7': r'D:/models/wave2_m0s7_s097/output_dir',
 'wave2_s090': r'D:/models/wave2_m0s1_s090/output_dir',
}
# 参考表: PIL 生成的 q 表
ref = {}
for q in range(60, 101):
    b = io.BytesIO()
    Image.new('RGB', (16, 16), (128, 100, 60)).save(b, format='JPEG', quality=q)
    im = Image.open(io.BytesIO(b.getvalue()))
    ref[q] = (tuple(im.quantization[0]), tuple(im.quantization[1]))

for k, d in DIRS.items():
    p = Path(d) / 'case1.jpg'
    if not p.exists():
        print(f'{k:12s} NO FILE'); continue
    im = Image.open(p)
    qt = (tuple(im.quantization[0]), tuple(im.quantization[1]))
    match = [q for q, v in ref.items() if v == qt]
    sub = im.layer if hasattr(im, 'layer') else None
    print(f'{k:12s} q={match if match else "?"}  luma[0..3]={qt[0][:4]}  size={im.size}')
