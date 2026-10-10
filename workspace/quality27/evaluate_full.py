import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
from PIL import Image
import torch
from common import ROOT,guard,digest
from infer import load_runner,render

VAL=Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
TEST=VAL.parent/'测试集'
MD={'v3':Path('D:/models/review_V3_20260926/V3_Review_Final/model_dir'),
    'v4t1':Path('D:/models/finals_20260927/Track2_light_diffusion_inference/model_dir'),
    'candidate':ROOT/'candidate/model_dir'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['val','nr'],required=True)
    ap.add_argument('--research-only',action='store_true');a=ap.parse_args()
    guard()
    export_pass=json.loads((ROOT/'export_check.json').read_text())['pass']
    if not export_pass and not (a.research_only and a.mode=='val'):raise RuntimeError('Export gate not passed')
    import lpips
    metrics={}
    if a.mode=='val':metrics['lpips']=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    else:
        prior=json.loads((ROOT/'full_val.json').read_text())
        if not prior['gate']:raise RuntimeError('Full validation gate failed; no test30 tuning')
        import pyiqa
        metrics={k:pyiqa.create_metric(k,device='cuda') for k in ['clipiqa','niqe']}
    rows=[];calls=0;metric_calls=0
    for tag,md in MD.items():
        guard();r=load_runner(md);calls+=4
        outdir=ROOT/('eval_'+a.mode)/tag;outdir.mkdir(parents=True,exist_ok=False)
        files=sorted(VAL.glob('*_lq.*')) if a.mode=='val' else sorted(TEST.glob('*.jpg'))[:30]
        def before():
            nonlocal calls
            guard();calls+=1
            if calls>12000:raise RuntimeError('Render budget exhausted')
        for f in files:
            img=np.array(Image.open(f).convert('RGB'))
            out=render(r,img,before)
            p=outdir/(f.stem+'.jpg');Image.fromarray(out).save(p,quality=98,subsampling=0)
            decoded=np.array(Image.open(p).convert('RGB'))
            t=torch.from_numpy(decoded.copy()).permute(2,0,1)[None].cuda().float()/255
            row={'model':tag,'input':f.name,'output_sha256':digest(p),'model_sha256':digest(md/'your_model.pt')}
            with torch.no_grad():
                if a.mode=='val':
                    gpath=next(VAL.glob(f.stem.replace('_lq','_gt')+'.*'))
                    gt=np.array(Image.open(gpath).convert('RGB'))
                    mse=np.mean((decoded.astype(np.float64)-gt.astype(np.float64))**2)
                    row['psnr']=float(10*np.log10(255**2/max(mse,1e-12)))
                    g=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda().float()/255
                    row['lpips']=float(metrics['lpips'](t*2-1,g*2-1));metric_calls+=1
                    del g
                else:
                    for name,m in metrics.items():row[name]=float(m(t));metric_calls+=1
            rows.append(row);del t
            print(json.dumps(row),flush=True)
            (ROOT/('full_'+a.mode+'_partial.json')).write_text(json.dumps({'rows':rows,'render_calls':calls,'metric_calls':metric_calls},indent=2))
        del r;torch.cuda.empty_cache()
    names=['psnr','lpips'] if a.mode=='val' else ['clipiqa','niqe']
    mean={tag:{key:float(np.mean([r[key] for r in rows if r['model']==tag])) for key in names} for tag in MD}
    cand=mean['candidate'];gate=True;checks={}
    for tag in ['v3','v4t1']:
        baseline=mean[tag]
        if a.mode=='val':
            c=(cand['psnr']>=baseline['psnr']-.15 and cand['lpips']<=baseline['lpips']*.98)
            per={r['input']:r['lpips'] for r in rows if r['model']==tag}
            c=c and all(r['lpips']<=per[r['input']]*1.05 for r in rows if r['model']=='candidate')
        else:c=cand['clipiqa']>=baseline['clipiqa']-.01 and cand['niqe']<=baseline['niqe']*1.03
        checks[tag]=bool(c);gate=gate and c
    report={'rows':rows,'mean':mean,'checks':checks,'gate':bool(gate),'export_gate':export_pass,
            'research_only':a.research_only,'delivery_eligible':bool(gate and export_pass and not a.research_only),
            'render_calls':calls,'metric_calls':metric_calls,
            'scope':'all compared models use the same repaired feather/rounding renderer; not an official score prediction'}
    (ROOT/('full_'+a.mode+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps({'mean':mean,'gate':bool(gate),'checks':checks}),flush=True)


if __name__=='__main__':main()
