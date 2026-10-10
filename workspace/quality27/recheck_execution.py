"""One bounded execution-mode repair; never overwrite failed export evidence."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
from PIL import Image
import torch
from common import ROOT,PARENT,Refiner,digest,guard
from export_model import Combined


def main():
    guard()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ck=ROOT/'conditional_s2000.pt'
    parent=torch.jit.load(str(PARENT),map_location='cuda').eval()
    ref=Refiner(True);ref.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['net'])
    model=Combined(parent,ref.cuda().half().eval()).cuda().half().eval()
    mpath=ROOT/'candidate/model_dir/your_model.pt'
    loaded=torch.jit.load(str(mpath),map_location='cuda').eval()
    torch.manual_seed(42)
    sample=np.array(Image.open('D:/models/review_V3_20260926/V3_Review_Final/samples/case3_in.jpg').convert('RGB'))[:512,:512]
    inputs=[torch.zeros(1,3,512,512,device='cuda',dtype=torch.float16),
            (torch.rand(1,3,512,512,device='cuda')*2-1).half(),
            (torch.from_numpy(sample.copy()).permute(2,0,1)[None].float()/127.5-1).cuda().half()]
    result={'model_sha256':digest(mpath),'execution':'optimized_execution(False), deterministic=True, TF32=False',
            'forward_calls_reserved':9,'reads':[],'pass':True}
    (ROOT/'execution_recheck_reserved.json').write_text(json.dumps(result,indent=2))
    with torch.no_grad(),torch.jit.optimized_execution(False):
        for i,x in enumerate(inputs):
            guard();e=model(x);a=loaded(x);b=loaded(x)
            d=(torch.round((e.float()+1)*127.5)-torch.round((a.float()+1)*127.5)).abs()
            r={'input':i,'finite':bool(torch.isfinite(e).all() and torch.isfinite(a).all()),
               'u8max':float(d.max()),'different_fraction':float((d>0).float().mean()),
               'repeat_exact':bool(torch.equal(a,b))}
            result['reads'].append(r)
            result['pass']=result['pass'] and r['finite'] and r['u8max']<=1 and r['different_fraction']<=.001 and r['repeat_exact']
    (ROOT/'execution_recheck.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
