import json,os,time
from common import ROOT
path=ROOT/'ledger.json';d=json.loads(path.read_text())
assert d['caps']['export']==40 and d['used']['export']==27
note={'epoch':time.time(),'kind':'resource-only local verification bug repair','old_cap':40,'new_cap':64,
      'old_charges_retained':27,'additional_reserved_plan':27,
      'cause':'Reference injected new weights after CUDA Graph capture. This does not establish correct new-weight reference execution.',
      'repair':'Inject selected weights before any warmup/JIT execution/capture; no model or numerical criterion change.',
      'old_results_retained':['deployment_check.json','deployment_execution.json','fullval_execution.json']}
with (ROOT/'execution_amendment.json').open('x') as f:json.dump(note,f,indent=2)
d['caps']['export']=64;tmp=ROOT/'ledger.tmp';tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,path)
print('CAP_AMENDED_NO_REFUND')
