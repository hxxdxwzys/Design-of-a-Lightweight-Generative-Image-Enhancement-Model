import argparse,json,os,sys,time,random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TORCH_HOME',str(Path.home()/'.cache/torch'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from common import ROOT,LQ,TGT,BASE,initial_ref,PatchD,guard,digest,PARENT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--resume',default='')
    a=ap.parse_args()
    if a.steps not in [2000,6000]:raise ValueError('Unregistered step count')
    guard()
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    if digest(PARENT)!=manifest['parent_sha256']:raise RuntimeError('Parent drift')
    torch.manual_seed(270927);random.seed(270927)
    rng=np.random.RandomState(270927)
    G=initial_ref(True).cuda();D=PatchD().cuda()
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    for p in lp.parameters():p.requires_grad_(False)
    optG=torch.optim.AdamW(G.parameters(),lr=5e-5,betas=(.9,.99))
    optD=torch.optim.AdamW(D.parameters(),lr=2e-4,betas=(.9,.99))
    sg=torch.amp.GradScaler('cuda');sd=torch.amp.GradScaler('cuda')
    start=0; prior_elapsed=0
    if a.resume:
        r=torch.load(a.resume,map_location='cpu',weights_only=True)
        G.load_state_dict(r['net']);D.load_state_dict(r['D']);optG.load_state_dict(r['optG']);optD.load_state_dict(r['optD'])
        sg.load_state_dict(r['sg']);sd.load_state_dict(r['sd']);start=r['steps'];prior_elapsed=r['training_wall_s']
        rng.set_state((r['rng_name'],np.array(r['rng_keys'],dtype=np.uint32),r['rng_pos'],r['rng_has_gauss'],r['rng_cached_gaussian']))
    train=manifest['train'];pool=ThreadPoolExecutor(max_workers=8)
    log=ROOT/('train_%d.jsonl'%a.steps)
    if log.exists():raise FileExistsError(str(log))
    t0=time.monotonic()
    def sample(spec):
        name,y,x,flip=spec
        out=[]
        for d in [BASE,LQ,TGT]:
            with Image.open(d/name) as im:
                ar=np.array(im.convert('RGB'))[y:y+256,x:x+256]
            if flip:ar=ar[:,::-1]
            out.append(ar.copy())
        return out
    print('TRAIN_START',a.steps,'from',start,flush=True)
    for step in range(start+1,a.steps+1):
        guard()
        if prior_elapsed+time.monotonic()-t0>5400:raise RuntimeError('Training cumulative 90min exhausted')
        specs=[(train[rng.randint(len(train))],rng.randint(257),rng.randint(257),bool(rng.randint(2))) for _ in range(4)]
        samples=list(pool.map(sample,specs))
        base,lq,tgt=[torch.from_numpy(np.stack([s[k] for s in samples])).permute(0,3,1,2).cuda().float()/127.5-1 for k in range(3)]
        with torch.autocast('cuda',dtype=torch.float16):
            fake=G(base,lq)
            dr=D(tgt);df=D(fake.detach())
            lossD=.5*(F.mse_loss(dr,torch.ones_like(dr))+F.mse_loss(df,torch.zeros_like(df)))
        if not torch.isfinite(lossD):raise RuntimeError('Nonfinite discriminator loss')
        optD.zero_grad(set_to_none=True);sd.scale(lossD).backward();sd.step(optD);sd.update()
        for p in D.parameters():p.requires_grad_(False)
        with torch.autocast('cuda',dtype=torch.float16):
            l1=F.l1_loss(fake.float(),tgt);per=lp(fake.float(),tgt).mean()
            pf=D(fake);gan=F.mse_loss(pf,torch.ones_like(pf));lossG=l1+.2*per+.1*gan
        if not torch.isfinite(lossG):raise RuntimeError('Nonfinite generator loss')
        optG.zero_grad(set_to_none=True);sg.scale(lossG).backward();sg.step(optG);sg.update()
        for p in D.parameters():p.requires_grad_(True)
        # Schedule fixed for the whole 6000-step plan, never restarted on resume.
        lr=5e-5*(.05+.95*(1+np.cos(np.pi*step/6000))/2)
        for group in optG.param_groups:group['lr']=float(lr)
        if step%100==0 or step==a.steps:
            r={'step':step,'l1':float(l1),'lpips':float(per),'G':float(lossG),'D':float(lossD),
               'wall_s':prior_elapsed+time.monotonic()-t0,'max_vram':torch.cuda.max_memory_allocated(),
               'G_training_forwards':step,'D_training_forwards':step*3,'lpips_training_forwards':step}
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
            print(json.dumps(r),flush=True)
        if step==a.steps:
            state=rng.get_state()
            torch.save({'net':G.cpu().state_dict(),'D':D.cpu().state_dict(),'optG':optG.state_dict(),'optD':optD.state_dict(),
                        'sg':sg.state_dict(),'sd':sd.state_dict(),'steps':step,'training_wall_s':prior_elapsed+time.monotonic()-t0,
                        'rng_name':state[0],'rng_keys':state[1].tolist(),'rng_pos':state[2],'rng_has_gauss':state[3],'rng_cached_gaussian':state[4],
                        'parent_sha256':manifest['parent_sha256'],'source_manifest_sha256':digest(ROOT/'data_manifest.json')},ROOT/('conditional_s%d.pt'%step))
    pool.shutdown()
    print('TRAIN_DONE',flush=True)


if __name__=='__main__':main()
