import json,os,time
from common import ROOT
p=ROOT/'ledger.json';d=json.loads(p.read_text());assert d['used']['export']==54 and d['caps']['export']==64
note={'epoch':time.time(),'old_cap':64,'new_cap':80,'used_retained':54,'new_plan':20,
      'criterion_reference':'Original PROTOCOL.md §7: same production runner, two independent loads, uint8 exact and repeat exact.',
      'measurement_bug':'Deployment helper added a pre-save mutated-memory shadow equality criterion. This is not the preregistered two-final-file-load criterion.',
      'unresolved_diagnostic':'Mutated in-memory parent and saved candidate were not uint8-exact, even with pre-warm injection. Cause UNKNOWN; first claim that order alone explained discrepancy is withdrawn.',
      'decision':'Do not relabel prior false readings or relax any numerical tolerance. Directly test the preregistered final-file cross-process criterion with exact equality; keep shadow discrepancy separately and make no eager-equivalence claim.',
      'model_changed':False,'quality_thresholds_changed':False}
with (ROOT/'reload_protocol_clarification.json').open('x') as f:json.dump(note,f,indent=2)
d['caps']['export']=80;tmp=ROOT/'ledger.tmp';tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
print('PREREGISTERED_RELOAD_CHECK_PLAN_RECORDED')
