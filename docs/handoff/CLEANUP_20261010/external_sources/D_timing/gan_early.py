# -*- coding: utf-8 -*-
"""GAN 中途 checkpoint 早读: 只出 24 张 4K, 量 lapvar/clipiqa, 判断"纹理档"有没有长出来"""
import argparse, os, sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--n', type=int, default=24)
a = ap.parse_args()
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(r'D:/tmp_timing/gan_early'); OUT.mkdir(exist_ok=True)
TILE, OVER, DOWN = 512, 32, 2
d = torch.load(a.ckpt, map_location='cpu', weights_only=False)
from diffusers import UNet2DModel
ch = tuple(d['ch']); alpha = float(d['alpha'])
net = UNet2DModel(sample_size=int(d['res']), in_channels=3, out_channels=3, layers_per_block=1,
                  block_out_channels=ch, down_block_types=('DownBlock2D',)*len(ch),
                  up_block_types=('UpBlock2D',)*len(ch))
net.load_state_dict(d['unet']); net = net.to('cuda').eval().half()
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r,:,0] *= ramp[:,None]; mask[-r:,:,0] *= ramp[::-1,None]
mask[:,:r,0] *= ramp[None,:]; mask[:,-r:,0] *= ramp[::-1][None,:]

@torch.no_grad()
def run(patch):
    t = torch.from_numpy(patch.copy()).permute(2,0,1)[None].to('cuda').float().div_(127.5).sub_(1)
    tin = F.interpolate(t, size=(TILE//DOWN, TILE//DOWN), mode='bicubic', align_corners=False).half()
    pred = net(tin, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
    y = (tin - sq_1a * pred.float()) / sq_a
    y = F.interpolate(y, size=(TILE, TILE), mode='bicubic', align_corners=False)
    return np.clip((y[0].permute(1,2,0).float().cpu().numpy()+1)*127.5, 0, 255)

names = [f'case{i}' for i in list(range(1, 101, 4))[:a.n]]
lv = []
for n in names:
    img = np.asarray(Image.open(TEST/f'{n}.jpg').convert('RGB'), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H,W,3), np.float32); ws = np.zeros((H,W,1), np.float32)
    for y in sorted(set(list(range(0, max(1,H-TILE+1), TILE-OVER)) + [max(0,H-TILE)])):
        for x in sorted(set(list(range(0, max(1,W-TILE+1), TILE-OVER)) + [max(0,W-TILE)])):
            acc[y:y+TILE, x:x+TILE] += run(img[y:y+TILE, x:x+TILE]) * mask
            ws[y:y+TILE, x:x+TILE] += mask
    out = np.clip(acc/np.maximum(ws,1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(OUT/f'{n}.jpg', quality=98)
    g = np.asarray(Image.fromarray(out).convert('L'), dtype=np.float32)
    lv.append(float(ndimage.laplace(g).var()))
print(f'lapvar = {np.mean(lv):.1f}  (参照: M24 508 / TinyV2 110 / v5 92 / LQ 38)', flush=True)
m = pyiqa.create_metric('clipiqa', device='cuda')
vs = []
for n in names:
    t = torch.from_numpy(np.asarray(Image.open(OUT/f'{n}.jpg').convert('RGB'))).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    with torch.no_grad(): vs.append(float(m(t).item()))
    del t
print(f'clipiqa = {np.mean(vs):.4f}  (参照: M24 0.9023 / v5 0.3897 / TinyV2 0.3532 / LQ 0.2752)', flush=True)
print('GAN_EARLY_DONE')
