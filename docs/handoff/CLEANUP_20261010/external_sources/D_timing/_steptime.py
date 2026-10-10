# -*- coding: utf-8 -*-
import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = Path('D:/models/tiny_v1/train_gan.log')
lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
pat = re.compile(r'^\[(\d\d:\d\d:\d\d)\] step (\d+)/25000 .*?([\d.]+)s/step')
rows = []
for ln in lines:
    m = pat.match(ln.strip())
    if m:
        rows.append((m.group(1), int(m.group(2)), float(m.group(3))))
print('v11b 步时演化 (每 1000 步的窗口均值, 用相邻记录的实际间隔算):')
prev = None
buckets = {}
for t, s, sp in rows:
    if prev:
        pt, ps = prev
        hh, mm, ss = (int(x) for x in t.split(':'))
        ph, pm, pss = (int(x) for x in pt.split(':'))
        dt = (hh * 3600 + mm * 60 + ss) - (ph * 3600 + pm * 60 + pss)
        if dt >= 0 and s > ps:
            b = (s // 1000) * 1000
            buckets.setdefault(b, []).append(dt / (s - ps))
    prev = (t, s)
print('%-18s %s' % ('step 窗口', '实测 s/step (窗口均值)'))
for b in sorted(buckets):
    v = buckets[b]
    print('  %5d-%5d      %.3f   (n=%d)' % (b, b + 1000, sum(v) / len(v), len(v)))
last = rows[-1] if rows else None
print('最后一条:', last)
