"""Preregistered final-file independent-load exactness; each invocation is a fresh process."""
import argparse,hashlib,importlib.util,json,os
import numpy as np
from PIL import Image
import torch
from common import ROOT,reserve,tensor,u8,finite,sha
from validate import modeldir


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',type=int,choices=[1,2],required=True);a=ap.parse_args()
    s=json.loads((ROOT/'selection.json').read_text())['steps'];md=modeldir(s)
    spec=importlib.util.spec_from_file_location('fresh_runner',str(md/'runner.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reserve('export',4);r=m.Runner(str(md));torch.manual_seed(42)
    name=json.loads((ROOT/'data_manifest.json').read_text())['rows'][0]['name']
    xs=[torch.zeros(1,3,512,512),torch.rand(1,3,512,512)*2-1,tensor(np.array(Image.open(ROOT/'data/lq'/(name+'.png')).convert('RGB')),'cpu')]
    rows=[];arrays=[]
    with torch.no_grad():
        for i,x in enumerate(xs):
            reserve('export');y=u8(finite(r.infer(x,prompt=''))).cpu()
            reserve('export');z=u8(finite(r.infer(x,prompt=''))).cpu()
            rows.append({'input':i,'repeat_equal':bool(torch.equal(y,z)),'sha256':hashlib.sha256(y.numpy().tobytes()).hexdigest()});arrays.append(y)
    torch.save(arrays,ROOT/('reload_%d_arrays.pt'%a.id))
    result={'pid':os.getpid(),'rows':rows,'model_sha256':sha(md/'your_model.pt'),'runner_sha256':sha(md/'runner.py'),
            'gate':all(row['repeat_equal'] for row in rows)}
    with (ROOT/('reload_%d.json'%a.id)).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
