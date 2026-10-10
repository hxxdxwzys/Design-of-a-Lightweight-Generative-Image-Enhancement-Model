# -*- coding: utf-8 -*-
"""0.5B runner 的 torch<2.0 兼容 patch (load 语句中间有空行)"""
from pathlib import Path

p = Path(r'D:/models/final_compliant/model_dir/runner.py')
lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
out = []
i = 0
patched = False
while i < len(lines):
    line = lines[i]
    if "pkg = torch.load(os.path.join(model_dir, 'your_model.pt')," in line and not patched:
        # 找对应的 weights_only 行
        j = i
        while j < len(lines) and 'weights_only=False)' not in lines[j]:
            j += 1
        out.append("        try:\n")
        out.append(line)
        for k in range(i + 1, j + 1):
            out.append(lines[k])
        out.append("        except TypeError:  # torch<2.0 无 weights_only 参数\n")
        out.append("            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')\n")
        i = j + 1
        patched = True
        continue
    out.append(line)
    i += 1
if patched:
    p.write_text(''.join(out), encoding='utf-8')
    print('0.5B runner patched')
else:
    print('未找到模式')
