import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from common import ROOT,PARENT,Refiner,digest,guard


class Combined(nn.Module):
    def __init__(self,parent,ref):
        super().__init__();self.unet=parent.unet;self.ref=ref
        self.register_buffer('tt',parent.tt.clone())
        self.register_buffer('sq_a',parent.sq_a.clone())
        self.register_buffer('sq_1a',parent.sq_1a.clone())
    def forward(self,x):
        xs=F.interpolate(x,size=(256,256),mode='bicubic',align_corners=False)
        pred=self.unet(self.tt,xs)
        y=(xs-self.sq_1a.to(x.dtype)*pred)/self.sq_a.to(x.dtype)
        base=F.interpolate(y,size=(512,512),mode='bicubic',align_corners=False)
        out=self.ref(base,x)
        ar=torch.arange(9,dtype=torch.float64,device=x.device)-4
        k=torch.exp(-(ar*ar)/(2*1.2*1.2));k=(k/k.sum()).to(x.dtype)
        blur=F.conv2d(F.pad(out,(4,4,0,0),mode='reflect'),k.view(1,1,1,-1).expand(3,1,1,-1),groups=3)
        blur=F.conv2d(F.pad(blur,(0,0,4,4),mode='reflect'),k.view(1,1,-1,1).expand(3,1,-1,1),groups=3)
        diff=out-blur;mask=(diff.abs()>2/255).to(x.dtype)
        out=(out+.6*diff*mask).clamp(-1,1)
        return torch.round((out+1)*127.5)/127.5-1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,required=True);a=ap.parse_args()
    guard();md=ROOT/'candidate/model_dir';md.mkdir(parents=True,exist_ok=False)
    ck=ROOT/('conditional_s%d.pt'%a.steps);d=torch.load(ck,map_location='cpu',weights_only=True)
    if d['parent_sha256']!=digest(PARENT):raise ValueError('Parent mismatch')
    parent=torch.jit.load(str(PARENT),map_location='cuda').eval()
    ref=Refiner(True);ref.load_state_dict(d['net']);ref=ref.cuda().eval().half()
    model=Combined(parent,ref).cuda().eval().half()
    torch.manual_seed(42)
    sample=np.array(Image.open('D:/models/review_V3_20260926/V3_Review_Final/samples/case3_in.jpg').convert('RGB'))[:512,:512]
    inputs=[torch.zeros(1,3,512,512,device='cuda',dtype=torch.float16),
            (torch.rand(1,3,512,512,device='cuda')*2-1).half(),
            (torch.from_numpy(sample.copy()).permute(2,0,1)[None].float()/127.5-1).cuda().half()]
    report={'steps':a.steps,'checkpoint_sha256':digest(ck),'parent_sha256':digest(PARENT),'reads':[],
            'export_forward_reserved':13,'new_model':'conditional residual refiner; unchanged inherited one-step UNet'}
    (ROOT/'export_reserved.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with torch.no_grad():
        traced=torch.jit.trace(model,inputs[0],check_trace=False,strict=False)
        traced.save(str(md/'your_model.pt'))
        reloaded=torch.jit.load(str(md/'your_model.pt'),map_location='cuda').eval()
        good=True
        for i,x in enumerate(inputs):
            eager=model(x);a1=reloaded(x);a2=reloaded(x)
            eu=torch.round((eager.float()+1)*127.5).clamp(0,255)
            ru=torch.round((a1.float()+1)*127.5).clamp(0,255)
            diff=(eu-ru).abs()
            r={'input':i,'finite':bool(torch.isfinite(eager).all() and torch.isfinite(a1).all()),
               'u8max':float(diff.max()),'different_fraction':float((diff>0).float().mean()),
               'repeat_exact':bool(torch.equal(a1,a2))}
            good=good and r['finite'] and r['u8max']<=1 and r['different_fraction']<=.001 and r['repeat_exact']
            report['reads'].append(r)
    shutil.copy2(PARENT.parent/'runner.py',md/'runner.py')
    report['pass']=bool(good);report['model_sha256']=digest(md/'your_model.pt');report['runner_sha256']=digest(md/'runner.py')
    report['params']=sum(p.numel() for p in reloaded.parameters())
    (ROOT/'export_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not good:raise RuntimeError('Export equivalence gate failed')


if __name__=='__main__':main()
