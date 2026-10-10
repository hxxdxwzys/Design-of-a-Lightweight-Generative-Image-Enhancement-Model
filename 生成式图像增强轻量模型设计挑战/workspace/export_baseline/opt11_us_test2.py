import io, os, sys, zipfile
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ZIP = r"D:\models\upload\Track2_TinyRefV1.zip"
PCT = float(sys.argv[1])
OUT = Path(r"D:\models\opt11\us_test%d" % int(PCT)); OUT.mkdir(parents=True, exist_ok=True)
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
names = [f.stem for f in sorted(TEST.glob("case*.jpg"))[:30]]
dev = "cuda"

def unsharp(t, sigma=1.2, percent=PCT, thr_u8=3.0):
    ksize = int(round(sigma*3))*2+1
    x = torch.arange(ksize, dtype=torch.float64) - (ksize-1)/2
    k = torch.exp(-(x*x)/(2.0*sigma*sigma)); k = (k/k.sum()).to(device=t.device, dtype=t.dtype)
    p = ksize//2; c = int(t.shape[1])
    y = F.pad(t, (p,p,0,0), mode="reflect"); y = F.conv2d(y, k.view(1,1,1,-1).expand(c,1,1,-1), groups=c)
    y = F.pad(y, (0,0,p,p), mode="reflect"); blur = F.conv2d(y, k.view(1,1,-1,1).expand(c,1,-1,1), groups=c)
    diff = t - blur; mask = (diff.abs() > thr_u8/255.0*2.0).to(t.dtype)
    out = (t + (percent/100.0)*diff*mask).clamp(-1,1)
    return (torch.round((out+1.0)*127.5)/127.5) - 1.0

with zipfile.ZipFile(ZIP) as zf:
    n = 0
    for nm in names:
        try: data = zf.read("output_dir/%s.jpg" % nm)
        except KeyError: continue
        a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(a.copy()).permute(2,0,1)[None].to(dev).float().div_(127.5).sub_(1)
        with torch.no_grad(): o = unsharp(t)
        Image.fromarray(((o[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)).save(OUT / ("%s.jpg" % nm), quality=98, subsampling=0)
        n += 1
print("WROTE", n, OUT, flush=True)
