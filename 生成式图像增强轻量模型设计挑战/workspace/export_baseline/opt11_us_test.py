r"""从已打包 zip 中取出 30 张（字典序）并测 unsharp 叠加效果（+ 反推是否需要）。"""
import os, sys, zipfile
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ZIP = r"D:\models\upload\Track2_TinyRefV1.zip"
OUT = Path(r"D:\models\opt11\us_test"); OUT.mkdir(parents=True, exist_ok=True)
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
names = [f.stem for f in sorted(TEST.glob("case*.jpg"))[:30]]      # lex 口径
dev = "cuda"

def unsharp(t, sigma=1.2, percent=60.0, thr_u8=3.0):
    ksize = int(round(sigma*3))*2+1
    x = torch.arange(ksize, dtype=torch.float64) - (ksize-1)/2
    k = torch.exp(-(x*x)/(2.0*sigma*sigma)); k = (k/k.sum()).to(device=t.device, dtype=t.dtype)
    p = ksize//2; c = int(t.shape[1])
    y = F.pad(t, (p,p,0,0), mode="reflect"); y = F.conv2d(y, k.view(1,1,1,-1).expand(c,1,1,-1), groups=c)
    y = F.pad(y, (0,0,p,p), mode="reflect"); blur = F.conv2d(y, k.view(1,1,-1,1).expand(c,1,-1,1), groups=c)
    diff = t - blur; mask = (diff.abs() > thr_u8/255.0*2.0).to(t.dtype)
    out = (t + (percent/100.0)*diff*mask).clamp(-1,1)
    u8 = torch.round((out+1.0)*127.5)
    return (u8/127.5) - 1.0

with zipfile.ZipFile(ZIP) as zf:
    got = 0
    for nm in names:
        member = "output_dir/%s.jpg" % nm
        try:
            data = zf.read(member)
        except KeyError:
            continue
        import io
        a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(a.copy()).permute(2,0,1)[None].to(dev).float().div_(127.5).sub_(1)
        with torch.no_grad():
            o = unsharp(t)
        arr = ((o[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(arr).save(OUT / ("%s.jpg" % nm), quality=98, subsampling=0)
        got += 1
print("US_TEST_WROTE", got, "->", OUT, flush=True)
