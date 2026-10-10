import argparse,json,os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
from PIL import Image
import torch
from common import ROOT,LQ,TGT,BASE,initial_ref,Refiner,guard,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,required=True);a=ap.parse_args()
    guard()
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    checkpoint=ROOT/('conditional_s%d.pt'%a.steps)
    d=torch.load(checkpoint,map_location='cpu',weights_only=True)
    candidate=Refiner(True).cuda().eval();candidate.load_state_dict(d['net'])
    parent=initial_ref(False).cuda().eval()
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    prior=None
    if a.steps==6000:prior=json.loads((ROOT/'validate_2000.json').read_text())
    readings=[]
    with torch.no_grad():
        for idx,name in enumerate(manifest['validation'][:32]):
            guard()
            vals=[]
            for folder in [BASE,LQ,TGT]:
                ar=np.array(Image.open(folder/name).convert('RGB'))[128:384,128:384]
                vals.append(torch.from_numpy(ar.copy()).permute(2,0,1)[None].cuda().float()/127.5-1)
            b,l,t=vals
            row={'name':name}
            for tag,model in [('parent',parent),('candidate',candidate)]:
                if tag=='parent' and prior is not None:
                    row[tag]=prior['readings'][idx]['parent'];continue
                out=model(b,l)
                if not torch.isfinite(out).all():raise RuntimeError('Nonfinite validation')
                row[tag]={'l1':float((out-t).abs().mean()),'lpips':float(lp(out,t).mean())}
            readings.append(row)
    mean={tag:{key:float(np.mean([r[tag][key] for r in readings])) for key in ['l1','lpips']} for tag in ['parent','candidate']}
    gate=mean['candidate']['l1']<=mean['parent']['l1'] and mean['candidate']['lpips']<=.98*mean['parent']['lpips']
    report={'steps':a.steps,'checkpoint_sha256':digest(checkpoint),'readings':readings,'mean':mean,'gate':bool(gate),
            'metric_calls':32 if prior else 64,'scope':'32 center crops from fine-tune holdout; no claim of unseen by parent'}
    (ROOT/('validate_%d.json'%a.steps)).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'steps':a.steps,'mean':mean,'gate':bool(gate)}),flush=True)


if __name__=='__main__':main()
