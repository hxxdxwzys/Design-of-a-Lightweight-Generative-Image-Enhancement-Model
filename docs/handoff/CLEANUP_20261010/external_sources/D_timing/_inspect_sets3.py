# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
M = Path(r'D:/models')
for d, sub in [('probe_a','pred_sc0.4'), ('probe_a','pred_sc0.4_jpg95'), ('probe_a','lq30')]:
    p = M / d / sub
    fs = sorted(p.glob('*'))
    print('%-24s n=%d first=%s' % (d + '/' + sub, len(fs), [f.name for f in fs[:4]]))
for d in ['wave2_m0s1_s090','wave2_m0s1_s097','wave2_m0s1_s100','wave2_m0s7_s097']:
    p = M / d
    print('---', d, 'val_gate:', (p/'val_gate.json').read_text(encoding='utf-8', errors='replace').replace('\n',' ')[:250])
    t = (p/'render.log').read_text(encoding='utf-8', errors='replace')
    keep = [ln for ln in t.splitlines() if ('sc' in ln.lower() and 'scale' in ln.lower()) or 'guidance' in ln.lower() or 'args' in ln.lower() or '--' in ln][:4]
    for k in keep: print('     ', k.strip()[:180])
