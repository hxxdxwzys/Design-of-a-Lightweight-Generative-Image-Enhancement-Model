import argparse,json,random,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from common import ROOT,initial_ref,reserve,guard,sha,EXPECTED


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,choices=[1000,3000],required=True);a=ap.parse_args()
    torch.manual_seed(92717);np.random.seed(92717);torch.set_num_threads(4)
    manifest=json.loads((ROOT/'data_manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='train']
    assert len(rows)==256
    allsrc={r['source'] for r in manifest['rows']};assert len(allsrc)==320
    for r in manifest['rows']:
        for kind in ['lq','gt','base']:
            path=ROOT/'data'/kind/(r['name']+('.pt' if kind=='base' else '.png'))
            if sha(path)!=r['files'][kind]:raise ValueError('Data fingerprint drift')
    base=torch.stack([torch.load(ROOT/'data/base'/(r['name']+'.pt'),weights_only=True) for r in rows])
    gt=torch.stack([torch.from_numpy(np.array(Image.open(ROOT/'data/gt'/(r['name']+'.png')).convert('RGB'))).permute(2,0,1) for r in rows])
    g=initial_ref().cuda().train();opt=torch.optim.Adam(g.parameters(),lr=1e-5);sc=torch.amp.GradScaler('cuda')
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    for p in lp.parameters():p.requires_grad_(False)
    start=0;elapsed=0
    if a.steps==3000:
        if not json.loads((ROOT/'dev_1000.json').read_text())['gate']:raise RuntimeError('Pilot gate FAIL')
        ck=torch.load(ROOT/'ref_s1000.pt',weights_only=True,map_location='cpu')
        g.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);sc.load_state_dict(ck['scaler']);start=1000;elapsed=ck['wall_s']
        torch.set_rng_state(ck['torch_rng'])
    log=ROOT/('train_%d.jsonl'%a.steps)
    if log.exists():raise FileExistsError(str(log))
    t0=time.monotonic()
    for step in range(start+1,a.steps+1):
        if elapsed+time.monotonic()-t0>3600:raise RuntimeError('Training wall cap')
        reserve('train');ix=torch.randint(0,len(rows),(4,));b=base[ix].cuda();target=gt[ix].cuda().float()/127.5-1
        with torch.autocast('cuda',dtype=torch.float16):
            out=g(b);l1=F.l1_loss(out.float(),target);per=lp(out.float(),target).mean()
            low=F.l1_loss(F.avg_pool2d(out.float(),8),F.avg_pool2d(target,8));loss=l1+.2*per+.1*low
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True);sc.scale(loss).backward();sc.step(opt);sc.update()
        if step%100==0:
            d={'step':step,'L1':float(l1),'LPIPS':float(per),'low':float(low),'loss':float(loss),'wall_s':elapsed+time.monotonic()-t0,
               'peak_vram':torch.cuda.max_memory_allocated(),'training_G_forwards':step,'training_LPIPS_forwards':step}
            with log.open('a') as f:f.write(json.dumps(d)+'\n')
            print(json.dumps(d),flush=True)
    torch.save({'state':{k:v.cpu() for k,v in g.state_dict().items()},'optimizer':opt.state_dict(),'scaler':sc.state_dict(),
                'torch_rng':torch.get_rng_state(),'steps':a.steps,'wall_s':elapsed+time.monotonic()-t0,'parent_sha256':EXPECTED,
                'data_manifest_sha256':sha(ROOT/'data_manifest.json')},ROOT/('ref_s%d.pt'%a.steps))
    print('TRAIN_DONE',a.steps,flush=True)


if __name__=='__main__':main()
