"""Same-subject tile renderer with repaired exterior feather weights; no image retrieval."""
from pathlib import Path
import argparse,importlib.util
import numpy as np
import torch
from PIL import Image

TILE,OVER=512,32


def positions(n):
    if n<TILE:raise ValueError('Both dimensions must be >=512')
    return sorted(set(list(range(0,n-TILE+1,TILE-OVER))+[n-TILE]))


def mask_at(y,x,H,W):
    a=np.ones(TILE,np.float32);b=np.ones(TILE,np.float32)
    r=OVER//2;v=np.linspace(0,1,r,dtype=np.float32)
    if y>0:a[:r]*=v
    if y+TILE<H:a[-r:]*=v[::-1]
    if x>0:b[:r]*=v
    if x+TILE<W:b[-r:]*=v[::-1]
    return (a[:,None]*b[None,:])[:,:,None]


def load_runner(model_dir):
    if not torch.cuda.is_available():raise RuntimeError('This inherited TorchScript requires CUDA; CPU inference is unsupported')
    spec=importlib.util.spec_from_file_location('quality27_runner',str(Path(model_dir)/'runner.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Runner(str(model_dir))


@torch.no_grad()
def render(runner,img,before_call=None):
    H,W=img.shape[:2]
    acc=np.zeros((H,W,3),np.float32);ws=np.zeros((H,W,1),np.float32)
    for y in positions(H):
        for x in positions(W):
            patch=img[y:y+TILE,x:x+TILE]
            t=torch.from_numpy(patch.copy()).permute(2,0,1)[None].float()/127.5-1
            if before_call:before_call()
            pred=runner.infer(t,prompt='')
            if pred.shape!=(1,3,TILE,TILE) or not torch.isfinite(pred).all():raise RuntimeError('Invalid tile')
            p=pred[0].permute(1,2,0).float().cpu().numpy()
            m=mask_at(y,x,H,W)
            acc[y:y+TILE,x:x+TILE]+=np.clip((p+1)*127.5,0,255)*m
            ws[y:y+TILE,x:x+TILE]+=m
    if np.any(ws<=0):raise RuntimeError('Uncovered image pixels')
    return np.rint(np.clip(acc/ws,0,255)).astype(np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--modeldir',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();src=Path(a.input);dst=Path(a.output)
    dst.mkdir(parents=True,exist_ok=True)
    files=[src] if src.is_file() else sorted(src.glob('*.jpg'))
    if not files:raise ValueError('No input images')
    if any((dst/f.name).exists() for f in files):raise FileExistsError('Refusing to overwrite existing outputs')
    # In a delivered review package, verify before executing either model or runner.
    if (Path(__file__).parent/'manifest.json').exists():
        from selfcheck import verify_identity
        verify_identity(Path(__file__).parent)
        if Path(a.modeldir).resolve()!=(Path(__file__).parent/'model_dir').resolve():raise ValueError('Use package model_dir')
    r=load_runner(a.modeldir)
    for f in files:
        p=dst/f.name
        if p.exists():raise FileExistsError(str(p))
        out=render(r,np.array(Image.open(f).convert('RGB')))
        Image.fromarray(out).save(p,quality=98,subsampling=0)
        print(f.name,flush=True)


if __name__=='__main__':main()
