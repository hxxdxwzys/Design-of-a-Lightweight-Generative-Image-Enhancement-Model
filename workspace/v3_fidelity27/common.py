import hashlib,json,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path('D:/models/v3_fidelity27_20260927')
PARENT=Path('D:/models/review_V3_20260926/V3_Review_Final/model_dir/your_model.pt')
EXPECTED='421be231d504b1e1d20656ff7aa1c45a377434ff4762cc79c05416a2c7b5306d'
DEADLINE=datetime(2026,9,27,11,30,tzinfo=timezone.utc).timestamp()
CAPS={'cache':400,'train':3000,'infer':10000,'metrics':350,'export':80,'bench':250}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def guard():
    if time.time()>=DEADLINE:raise RuntimeError('19:30 hard research deadline')
    if shutil.disk_usage(ROOT.parent).free<4*1024**3:raise RuntimeError('Free disk <4GiB')


def reserve(kind,n=1):
    guard();p=ROOT/'ledger.json';d=json.loads(p.read_text())
    if kind not in CAPS or n<1 or d['used'][kind]+n>d['caps'][kind]:raise RuntimeError('Budget exceeded: '+kind)
    d['used'][kind]+=n
    tmp=ROOT/'ledger.tmp';tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)


def parent(device='cpu'):
    if sha(PARENT)!=EXPECTED:raise ValueError('V3 parent fingerprint changed')
    return torch.jit.load(str(PARENT),map_location=device).eval()


class Refiner(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,16,3,1,1),nn.ReLU(),nn.Conv2d(16,16,3,1,1),nn.ReLU(),
                                                 nn.Conv2d(16,16,3,1,1),nn.ReLU(),nn.Conv2d(16,3,3,1,1))
    def forward(self,x):return (x+self.net(x)).clamp(-1,1)


def initial_ref():
    p=parent();g=Refiner();g.load_state_dict({k:v.float() for k,v in p.ref.state_dict().items()});return g


@torch.no_grad()
def student(p,x):
    xs=F.interpolate(x,size=(256,256),mode='bicubic',align_corners=False)
    e=p.unet(p.tt,xs)
    y=(xs-p.sq_1a.to(x.dtype)*e)/p.sq_a.to(x.dtype)
    return F.interpolate(y,size=(512,512),mode='bicubic',align_corners=False)


def tensor(a,device='cuda'):
    return (torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/127.5-1).to(device)


def u8(y):return torch.round((y.float()+1)*127.5).clamp(0,255).byte()


def finite(x):
    if not torch.isfinite(x).all():raise ValueError('Nonfinite tensor')
    return x
