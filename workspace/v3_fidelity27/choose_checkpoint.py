"""Choose once from dev results; avoid shadowing the stdlib select module."""
import json
from common import ROOT
a=json.loads((ROOT/'dev_1000.json').read_text());b=json.loads((ROOT/'dev_3000.json').read_text())
assert a['gate']
ma,mb=a['mean']['candidate'],b['mean']['candidate']
steps=3000 if b['gate'] and mb['lpips']<=ma['lpips'] and mb['psnr']>=ma['psnr'] else 1000
with (ROOT/'selection.json').open('x') as f:json.dump({'steps':steps,'basis':'dev only; both PSNR/LPIPS nonworse to select 3000',
        'dev1000':a['mean'],'dev3000':b['mean'],'holdout_unread':True,'official_val_unread_for_selection':True},f,indent=2)
print('SELECTED',steps)
