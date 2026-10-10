import argparse,importlib.util,io,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import ROOT,PARENT,reserve,sha,finite,tensor
from validate import runner,modeldir

VAL=Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
TEST=VAL.parent/'测试集'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['val','nr'],required=True);a=ap.parse_args()
    sel=json.loads((ROOT/'selection.json').read_text());steps=sel['steps']
    if not json.loads((ROOT/'reload_gate.json').read_text())['gate']:raise RuntimeError('Preregistered cross-process reload gate failed')
    if not json.loads((ROOT/('holdout_%d.json'%steps)).read_text())['gate']:raise RuntimeError('Holdout gate failed')
    if a.mode=='nr' and not json.loads((ROOT/'full_val.json').read_text())['gate']:raise RuntimeError('Full validation failed')
    f=Path('C:/生成式图像增强轻量模型设计挑战/workspace/quality27/infer.py')
    spec=importlib.util.spec_from_file_location('fidelity_renderer',str(f));render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    if a.mode=='val':
        import lpips
        metrics={'lpips':lpips.LPIPS(net='alex',verbose=False).cuda().eval()}
    else:
        import pyiqa
        metrics={k:pyiqa.create_metric(k,device='cuda') for k in ['clipiqa','niqe']}
    allrows=[]
    for tag,md in [('parent',PARENT.parent),('candidate',modeldir(steps))]:
        r=runner(md);dest=ROOT/('full_'+a.mode)/tag;dest.mkdir(parents=True,exist_ok=False)
        files=sorted(VAL.glob('*_lq.*')) if a.mode=='val' else sorted(TEST.glob('*.jpg'))[:30]
        for p in files:
            arr=np.array(Image.open(p).convert('RGB'))
            out=render.render(r,arr,lambda:reserve('infer'))
            target=dest/(p.stem+'.jpg');Image.fromarray(out).save(target,quality=98,subsampling=0)
            decoded=np.array(Image.open(target).convert('RGB'));row={'model':tag,'input':p.name,'output_sha256':sha(target),
                'input_sha256':sha(p),'model_sha256':sha(md/'your_model.pt'),'runner_sha256':sha(md/'runner.py'),'renderer_sha256':sha(f)}
            with torch.no_grad():
                if a.mode=='val':
                    gt=np.array(Image.open(next(VAL.glob(p.stem.replace('_lq','_gt')+'.*'))).convert('RGB'))
                    mse=np.mean((decoded.astype(np.float64)-gt.astype(np.float64))**2)
                    row['psnr']=float(10*np.log10(255**2/max(mse,1e-12)))
                    reserve('metrics');row['lpips']=float(finite(metrics['lpips'](tensor(decoded),tensor(gt))))
                else:
                    t=tensor(decoded)*.5+.5
                    for name,m in metrics.items():reserve('metrics');row[name]=float(finite(m(t)))
            allrows.append(row);print(json.dumps(row),flush=True)
            (ROOT/('full_'+a.mode+'_partial.json')).write_text(json.dumps(allrows,indent=2))
        del r;torch.cuda.empty_cache()
    keys=['psnr','lpips'] if a.mode=='val' else ['clipiqa','niqe']
    means={tag:{k:float(np.mean([r[k] for r in allrows if r['model']==tag])) for k in keys} for tag in ['parent','candidate']}
    b,c=means['parent'],means['candidate']
    if a.mode=='val':
        pmap={r['input']:r for r in allrows if r['model']=='parent'}
        gate=c['lpips']<=b['lpips']*.98 and c['psnr']>=b['psnr']-.1
        gate=gate and all(r['lpips']<=pmap[r['input']]['lpips']*1.05 for r in allrows if r['model']=='candidate')
    else:gate=c['clipiqa']>=b['clipiqa']-.01 and c['niqe']<=b['niqe']*1.03
    result={'rows':allrows,'mean':means,'gate':bool(gate),'renderer':'same fixed repaired feather/rounding for both',
            'no_retuning_on_these_results':True}
    (ROOT/('full_'+a.mode+'.json')).write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps(means),'GATE',gate,flush=True)


if __name__=='__main__':main()
