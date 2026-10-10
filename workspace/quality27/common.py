import hashlib
import json
import os
from pathlib import Path
import time
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path('D:/models/quality27_20260927')
LQ = Path('D:/datasets/distill256/lq')
TGT = Path('D:/datasets/distill256/tgt')
BASE = Path('D:/models/refine_v1/in')
PARENT = Path('D:/models/finals_20260927/Track2_light_diffusion_inference/model_dir/your_model.pt')
DEADLINE = 1790510400  # 2026-09-27 12:00 UTC / 20:00 Asia/Shanghai


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard():
    import shutil
    if time.time()>=DEADLINE: raise RuntimeError('Hard deadline reached')
    if shutil.disk_usage(ROOT.parent).free<4*1024**3: raise RuntimeError('Disk free <4GiB')


class Refiner(nn.Module):
    def __init__(self, conditional=True):
        super().__init__()
        self.conditional=conditional
        self.net=nn.Sequential(nn.Conv2d(6 if conditional else 3,16,3,1,1),nn.ReLU(),
                               nn.Conv2d(16,16,3,1,1),nn.ReLU(),
                               nn.Conv2d(16,16,3,1,1),nn.ReLU(),nn.Conv2d(16,3,3,1,1))
    def forward(self,base,lq):
        x=torch.cat((base,lq),dim=1) if self.conditional else base
        return (base+self.net(x)).clamp(-1,1)


def initial_ref(conditional=True):
    parent=torch.jit.load(str(PARENT),map_location='cpu')
    state={k:v.float() for k,v in parent.ref.state_dict().items()}
    model=Refiner(conditional)
    if conditional:
        w=state['net.0.weight']
        state['net.0.weight']=torch.cat((w,torch.zeros_like(w)),dim=1)
    model.load_state_dict(state,strict=True)
    return model


class PatchD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,64,4,2,1),nn.LeakyReLU(.2),
                               nn.Conv2d(64,128,4,2,1,bias=False),nn.BatchNorm2d(128),nn.LeakyReLU(.2),
                               nn.Conv2d(128,256,4,2,1,bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(.2),
                               nn.Conv2d(256,1,4,1,1))
    def forward(self,x):return self.net(x)
