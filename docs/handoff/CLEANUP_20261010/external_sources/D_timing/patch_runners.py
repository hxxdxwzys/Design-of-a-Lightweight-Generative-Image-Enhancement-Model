# -*- coding: utf-8 -*-
"""恢复并 patch runner (torch<2.0 兼容)"""
import zipfile
from pathlib import Path

z = zipfile.ZipFile(r'D:/models/Track2_14B_M0S1_SC093.zip')
good = z.read('model_dir/runner.py')
for f in ['final_compliant_14b', 'final_compliant_g3w18s1', 'final_compliant_m0s7']:
    Path(rf'D:/models/{f}/model_dir/runner.py').write_bytes(good)
    print('restored', f)

OLD = """        pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),
                         map_location='cpu', weights_only=False)"""
NEW = """        try:
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),
                             map_location='cpu', weights_only=False)
        except TypeError:  # torch<2.0 无 weights_only 参数
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')"""

for f in ['final_compliant_14b', 'final_compliant_g3w18s1', 'final_compliant_m0s7']:
    p = Path(rf'D:/models/{f}/model_dir/runner.py')
    c = p.read_text(encoding='utf-8')
    if OLD not in c:
        print(f'PATTERN MISSING in {f}')
        continue
    p.write_text(c.replace(OLD, NEW), encoding='utf-8')
    print('patched', f)

# 0.5B runner (final_compliant) 单独处理 (其缩进为 8 空格)
p05 = Path(r'D:/models/final_compliant/model_dir/runner.py')
c05 = p05.read_text(encoding='utf-8')
OLD05 = """        pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),
                         map_location='cpu', weights_only=False)"""
if OLD05 in c05:
    p05.write_text(c05.replace(OLD05, NEW), encoding='utf-8')
    print('patched final_compliant (0.5B)')
else:
    print('0.5B pattern missing; 检查其 torch.load 行')
    for i, line in enumerate(c05.splitlines(), 1):
        if 'torch.load' in line:
            print(f'  L{i}: {line!r}')
