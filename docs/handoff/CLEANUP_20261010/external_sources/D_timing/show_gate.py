import io
from pathlib import Path
s = Path(r'D:/tmp_timing/pixb_chain.log').read_text(encoding='utf-8', errors='replace').splitlines()
# 找最后一次门禁输出
idx = [i for i, l in enumerate(s) if 'Track2_PixB.zip' in l and ('门禁' in l or 'gate' in l.lower())]
start = idx[-1] if idx else max(0, len(s) - 45)
print('\n'.join(s[start:start + 40]))
