# -*- coding: utf-8 -*-
"""HYPIR 在 val 上的 512 裁切有参指标 (与 M24 / PixRestore-b 同口径)"""
import os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
HYPIR_REPO = r'D:/models/hypir_code/HYPIR-main/HYPIR-main'
sys.path.insert(0, HYPIR_REPO)
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner
import lpips as lpips_mod

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
OUT = Path(r'D:/tmp_timing/val_hypir'); OUT.mkdir(exist_ok=True)
LM = ('to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj')
print('[load] HYPIR ...', flush=True)
model = SD2Enhancer(base_model_path=r'D:/models/sd21base', weight_path=r'D:/models/hypir/HYPIR_sd2.pth',
                    lora_modules=LM.split(','), lora_rank=256, model_t=200, coeff_t=200, device='cuda')
model.init_models()
cap = EmptyCaptioner('cuda'); to_t = transforms.ToTensor()
lp = lpips_mod.LPIPS(net='alex', verbose=False)
def psnr(p, g):
    m = np.mean((p.astype(np.float64)-g.astype(np.float64))**2)
    return 99.0 if m <= 0 else 10*np.log10(255.0**2/m)
def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var(); cov = ((p-m1)*(g-m2)).mean()
    return ((2*m1*m2+C1)*(2*cov+C2))/((m1**2+m2**2+C1)*(v1+v2+C2))
res = []
for c in (1, 2, 3):
    lqim = Image.open(next(VAL.glob(f'case{c}_lq.*'))).convert('RGB')
    gtim = Image.open(next(VAL.glob(f'case{c}_gt.*'))).convert('RGB')
    W, H = lqim.size
    box = ((W-512)//2, (H-512)//2, (W-512)//2+512, (H-512)//2+512)
    lqc = lqim.crop(box); gtc = np.asarray(gtim.crop(box), dtype=np.uint8)
    t0 = time.time()
    with torch.no_grad():
        out = model.enhance(lq=to_t(lqc).unsqueeze(0).to('cuda'), prompt=cap(lqc),
                            scale_by='factor', upscale=1, patch_size=512, stride=512, return_type='pil')[0]
    oc = np.asarray(out.convert('RGB').resize((512,512), Image.BICUBIC), dtype=np.uint8)
    if c == 1:
        print(f'[latency] 512: {(time.time()-t0)*1000:.0f} ms -> {285.6/((time.time()-t0)*1000):.2f}x', flush=True)
    t2 = lambda z: torch.from_numpy(z.transpose(2,0,1)[None]/127.5-1).float()
    with torch.no_grad():
        l_p = float(lp(t2(oc), t2(gtc)).item())
    res.append((psnr(oc, gtc), ssim_np(oc, gtc), l_p))
    print(f'case{c}: HYPIR PSNR {res[-1][0]:.3f} | SSIM {res[-1][1]:.4f} | LPIPS {l_p:.4f}', flush=True)
a = np.array(res)
print(f'\nHYPIR 均值: PSNR {a[:,0].mean():.3f} | SSIM {a[:,1].mean():.4f} | LPIPS {a[:,2].mean():.4f}', flush=True)
print('同口径参照: LQ 23.953/0.9249/0.4983 | M24 22.615/0.9016/0.4408 | PixRestore-b 23.709/0.9199/0.4107', flush=True)
print('HYPIR_VAL_DONE')
