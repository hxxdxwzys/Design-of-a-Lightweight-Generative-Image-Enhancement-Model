import hashlib,importlib.util,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import ROOT,parent,student,reserve,sha,tensor,guard


def main():
    source=Path('D:/datasets/div2k/pngtxt_ori')
    files=sorted(source.glob('*.png'),key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())[:320]
    assert len(files)==320 and len({p.stem for p in files})==320
    script=Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/data/degradation.py')
    spec=importlib.util.spec_from_file_location('historical_degradation',str(script));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    out=ROOT/'data';out.mkdir(exist_ok=False)
    for d in ['lq','gt','base']:(out/d).mkdir()
    p=parent('cuda');rows=[];torch.set_num_threads(4)
    for i,f in enumerate(files):
        guard();seed=92717000+i;random.seed(seed);np.random.seed(seed)
        im=Image.open(f).convert('RGB');W,H=im.size
        if min(W,H)<512:raise ValueError('Unexpected source size')
        x=random.randint(0,W-512);y=random.randint(0,H-512);gt=im.crop((x,y,x+512,y+512));lq=m.degrade(gt)
        gt.save(out/'gt'/(f.stem+'.png'));lq.save(out/'lq'/(f.stem+'.png'))
        reserve('cache');base=student(p,tensor(np.array(lq)).half())
        if not torch.isfinite(base).all():raise ValueError('Nonfinite student output')
        torch.save(base[0].cpu().contiguous(),out/'base'/(f.stem+'.pt'))
        row={'name':f.stem,'split':'dev' if i<32 else ('holdout' if i<64 else 'train'),'source':str(f),
             'source_sha256':sha(f),'crop_xy':[x,y],'seed':seed,
             'files':{k:sha(out/k/(f.stem+('.pt' if k=='base' else '.png'))) for k in ['lq','gt','base']}}
        rows.append(row)
        if (i+1)%40==0:print('PREPARED',i+1,flush=True)
    # Measured legacy mismatch on fixed first eight names, not used as optimization data.
    legacy=[];names=sorted(Path('D:/datasets/distill256/lq').glob('*.jpg'))[:8]
    for f in names:
        reserve('cache');a=np.array(Image.open(f).convert('RGB'));b=student(p,tensor(a).half()).float().cpu()
        old=tensor(np.array(Image.open(Path('D:/models/refine_v1/in')/f.name).convert('RGB')),'cpu')
        dif=(b-old).abs();legacy.append({'name':f.name,'max':float(dif.max()),'mean':float(dif.mean()),'outside_unit_fraction':float((b.abs()>1).float().mean())})
    manifest={'rows':rows,'degradation_source':str(script),'degradation_sha256':sha(script),'legacy_cache_comparison':legacy,
              'scope':'fresh source-separated fine-tune split only; parent exposure unknown; existing source license history inherited'}
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DATA_READY',len(rows),'legacy_mean',sum(r['mean'] for r in legacy)/8,flush=True)


if __name__=='__main__':main()
