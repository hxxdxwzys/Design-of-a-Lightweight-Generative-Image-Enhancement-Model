import argparse,importlib.util,io,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import ROOT,PARENT,EXPECTED,parent,sha,tensor,u8,reserve,guard,finite


def modeldir(steps):
    md=ROOT/'candidates'/('s%d'%steps)/'model_dir'
    if md.exists():return md
    md.mkdir(parents=True,exist_ok=False)
    ck=torch.load(ROOT/('ref_s%d.pt'%steps),map_location='cpu',weights_only=True)
    if ck['parent_sha256']!=EXPECTED:raise ValueError('Parent differs')
    p=parent();before={k:v.clone() for k,v in p.state_dict().items() if not k.startswith('ref.')}
    graph=str(p.inlined_graph);p.ref.load_state_dict(ck['state'])
    unchanged=all(torch.equal(v,p.state_dict()[k]) for k,v in before.items())
    ref_ok=all(torch.equal(v.half(),p.ref.state_dict()[k]) for k,v in ck['state'].items())
    assert unchanged and ref_ok and str(p.inlined_graph)==graph
    p.save(str(md/'your_model.pt'));shutil.copy2(PARENT.parent/'runner.py',md/'runner.py')
    (md.parent/'conversion.json').write_text(json.dumps({'parent':EXPECTED,'model_sha256':sha(md/'your_model.pt'),
        'checkpoint_sha256':sha(ROOT/('ref_s%d.pt'%steps)),'nonref_exact':unchanged,'ref_exact':ref_ok,
        'graph_unchanged_in_memory':True,'retrace':False},indent=2))
    return md


def runner(md):
    spec=importlib.util.spec_from_file_location('fid_runner',str(md/'runner.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reserve('infer',4);return m.Runner(str(md))


def metrics(y,gt,lp):
    a=u8(finite(y))[0].permute(1,2,0).cpu().numpy();buf=io.BytesIO();Image.fromarray(a).save(buf,format='JPEG',quality=98,subsampling=0)
    out=np.array(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'));mse=np.mean((out.astype(np.float64)-gt.astype(np.float64))**2)
    reserve('metrics');l=float(finite(lp(tensor(out),tensor(gt))))
    return {'psnr':float(10*np.log10(255**2/max(mse,1e-12))),'lpips':l}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,choices=[1000,3000],required=True);ap.add_argument('--split',choices=['dev','holdout'],default='dev');a=ap.parse_args()
    if a.split=='holdout':
        selected=json.loads((ROOT/'selection.json').read_text());assert selected['steps']==a.steps
    reportfile=ROOT/('%s_%d.json'%(a.split,a.steps))
    if reportfile.exists():raise FileExistsError(str(reportfile))
    md=modeldir(a.steps)
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
    items=[r for r in json.loads((ROOT/'data_manifest.json').read_text())['rows'] if r['split']==a.split]
    cached=json.loads((ROOT/'dev_1000.json').read_text())['rows'] if a.steps==3000 and a.split=='dev' else None
    prior={r['name']:r['parent'] for r in cached} if cached else {}
    with torch.no_grad():
        if not prior:
            r=runner(PARENT.parent)
            for item in items:
                x=np.array(Image.open(ROOT/'data/lq'/(item['name']+'.png')).convert('RGB'))
                gt=np.array(Image.open(ROOT/'data/gt'/(item['name']+'.png')).convert('RGB'))
                reserve('infer');prior[item['name']]=metrics(r.infer(tensor(x,'cpu'),prompt=''),gt,lp)
            del r;torch.cuda.empty_cache()
        r=runner(md)
        for item in items:
            x=np.array(Image.open(ROOT/'data/lq'/(item['name']+'.png')).convert('RGB'))
            gt=np.array(Image.open(ROOT/'data/gt'/(item['name']+'.png')).convert('RGB'))
            reserve('infer');cur=metrics(r.infer(tensor(x,'cpu'),prompt=''),gt,lp)
            rows.append({'name':item['name'],'parent':prior[item['name']],'candidate':cur})
    means={tag:{k:float(np.mean([r[tag][k] for r in rows])) for k in ['psnr','lpips']} for tag in ['parent','candidate']}
    win=sum(r['candidate']['lpips']<r['parent']['lpips'] for r in rows)
    reg=sum(r['candidate']['lpips']>r['parent']['lpips']*1.10 for r in rows)
    gate=means['candidate']['lpips']<=means['parent']['lpips']*.98 and means['candidate']['psnr']>=means['parent']['psnr']-.10
    if a.split=='holdout':gate=gate and win>=20 and reg<=2
    report={'split':a.split,'steps':a.steps,'rows':rows,'mean':means,'lpips_wins':win,'large_regressions':reg,'gate':bool(gate),
            'model_sha256':sha(md/'your_model.pt'),'scope':'new synthetic degradation; fine-tune heldout sources, parent exposure UNKNOWN'}
    reportfile.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
