import io, sys
from pathlib import Path
BT = chr(96)
p = Path(r'C:\生成式图像增强轻量模型设计挑战\AGENTS.md')
t = io.open(p, encoding='utf-8').read()
n = t.count('%%BT%%')
io.open(p, 'w', encoding='utf-8').write(t.replace('%%BT%%', BT))
print('替换占位符:', n, '| 残留:', io.open(p, encoding='utf-8').read().count('%%BT%%'))
