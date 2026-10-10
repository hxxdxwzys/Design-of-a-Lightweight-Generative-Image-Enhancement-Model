import json
import torch
from common import ROOT
a=json.loads((ROOT/'reload_1.json').read_text());b=json.loads((ROOT/'reload_2.json').read_text())
x=torch.load(ROOT/'reload_1_arrays.pt',weights_only=True);y=torch.load(ROOT/'reload_2_arrays.pt',weights_only=True)
conversion=json.loads((ROOT/'candidates'/('s%d'%json.loads((ROOT/'selection.json').read_text())['steps'])/'conversion.json').read_text())
same=all(torch.equal(i,j) for i,j in zip(x,y)) and len(x)==len(y)==3
d={'gate':bool(a['gate'] and b['gate'] and a['pid']!=b['pid'] and same and a['model_sha256']==b['model_sha256']
              and conversion['nonref_exact'] and conversion['ref_exact'] and conversion['graph_unchanged_in_memory']),
   'cross_process_u8_exact':same,'different_pids':a['pid']!=b['pid'],'same_model':a['model_sha256']==b['model_sha256'],
   'model_sha256':a['model_sha256'],'conversion':conversion,
   'shadow_reference_discrepancy_retained':'deployment_check.json and deployment_check_r1.json; not claimed as eager equivalence'}
with (ROOT/'reload_gate.json').open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
