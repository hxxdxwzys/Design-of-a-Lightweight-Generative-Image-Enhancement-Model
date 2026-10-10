import json,importlib.util
import numpy as np
from PIL import Image
import torch
from unittest.mock import patch
from common import ROOT,PARENT,reserve,sha,u8,finite,tensor
from validate import modeldir


def load(md,state=None):
    spec=importlib.util.spec_from_file_location('deploy_runner',str(md/'runner.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reserve('export',4)
    if state is None:return m.Runner(str(md))
    original=torch.jit.load
    def injected(*args,**kwargs):
        p=original(*args,**kwargs);p.ref.load_state_dict(state);return p
    # Inject before eval, warmup, JIT optimization or CUDA Graph capture.
    with patch.object(torch.jit,'load',side_effect=injected):return m.Runner(str(md))


def main():
    s=json.loads((ROOT/'selection.json').read_text())['steps'];md=modeldir(s)
    if not json.loads((ROOT/('holdout_%d.json'%s)).read_text())['gate']:raise RuntimeError('Holdout failed')
    ck=torch.load(ROOT/('ref_s%d.pt'%s),map_location='cpu',weights_only=True)
    baseline=load(PARENT.parent,state=ck['state'])
    torch.manual_seed(42)
    f=ROOT/'data/lq'/ (json.loads((ROOT/'data_manifest.json').read_text())['rows'][0]['name']+'.png')
    xs=[torch.zeros(1,3,512,512),torch.rand(1,3,512,512)*2-1,tensor(np.array(Image.open(f).convert('RGB')),'cpu')]
    refs=[]
    with torch.no_grad():
        for x in xs:reserve('export');refs.append(u8(finite(baseline.infer(x,prompt=''))).cpu())
        del baseline;torch.cuda.empty_cache()
        reads=[]
        for process_load in range(2):
            r=load(md)
            for i,x in enumerate(xs):
                reserve('export');a=u8(finite(r.infer(x,prompt=''))).cpu()
                reserve('export');b=u8(finite(r.infer(x,prompt=''))).cpu()
                reads.append({'load':process_load,'input':i,'reference_equal':bool(torch.equal(a,refs[i])),
                              'repeat_equal':bool(torch.equal(a,b))})
            del r;torch.cuda.empty_cache()
    result={'reads':reads,'gate':all(r['reference_equal'] and r['repeat_equal'] for r in reads),
            'model_sha256':sha(md/'your_model.pt'),'runner_sha256':sha(md/'runner.py'),
            'scope':'same frozen V3 graph with copied ref weights before save vs two separate reloads in one process; independent-process full-image check required before delivery'}
    result['reference_weights_installed_before_warmup']=True
    with (ROOT/'deployment_check_r1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
