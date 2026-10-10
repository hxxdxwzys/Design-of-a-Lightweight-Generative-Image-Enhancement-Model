# -*- coding: utf-8 -*-
"""HYPIR 蒸馏数据生成 (2026-09-14 夜): (退化LQ, HYPIR输出) 成对 512 patch
动机: 排行榜显示新机制下 86~124 可达 => 差距在图像质量项; 而我们从没换过图像源.
      HYPIR 是已发表的 SOTA 复原模型(SD2.1 底), val LPIPS 0.3364 优于我们全家族.
"""
from __future__ import annotations
import io, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter

os.environ.setdefault('HF_HUB_OFFLINE', '1')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
HYPIR_REPO = r'D:/models/hypir_code/HYPIR-main/HYPIR-main'
sys.path.insert(0, HYPIR_REPO)
PATCH = sorted(Path(r'D:/datasets/tiny_patches').glob('*.jpg'))
OUT = Path(r'D:/datasets/distill_hypir')
(OUT / 'lq').mkdir(parents=True, exist_ok=True)
(OUT / 'tgt').mkdir(parents=True, exist_ok=True)
N = 2000
POOL = ThreadPoolExecutor(max_workers=8)


def degrade(arr_u8):
    im = Image.fromarray(arr_u8)
    q = int(np.random.uniform(80, 100))
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=q, subsampling=int(np.random.choice([0, 2])))
    buf.seek(0); im = Image.open(buf).convert('RGB')
    sig = min(abs(float(np.random.normal(0, 0.7))), 2.5)
    if sig > 0.05:
        im = im.filter(ImageFilter.GaussianBlur(radius=sig))
    a = np.asarray(im, dtype=np.float32)
    n = float(np.clip(math.exp(float(np.random.normal(math.log(1.05), 0.6))), 0.15, 3.5))
    return np.clip(a + np.random.standard_normal(a.shape).astype(np.float32) * n, 0, 255).astype(np.uint8)


def load_lq(_i):
    p = PATCH[np.random.randint(len(PATCH))]
    return degrade(np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from torchvision import transforms
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.captioner import EmptyCaptioner
    LM = ('to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,'
          'proj_in,proj_out,ff.net.2,ff.net.0.proj')
    t0 = time.time()
    print('[load] HYPIR SD2Enhancer ...', flush=True)
    model = SD2Enhancer(base_model_path=r'D:/models/sd21base',
                        weight_path=r'D:/models/hypir/HYPIR_sd2.pth',
                        lora_modules=LM.split(','), lora_rank=256,
                        model_t=200, coeff_t=200, device='cuda')
    model.init_models()
    cap = EmptyCaptioner('cuda')
    to_tensor = transforms.ToTensor()
    start = len(list((OUT / 'lq').glob('*.jpg')))
    print(f'[ready] {time.time()-t0:.0f}s | 已有 {start} 对, 目标 {N}', flush=True)
    i = start
    while i < N:
        lq = np.stack(list(POOL.map(load_lq, range(4))))[0]
        pil = Image.fromarray(lq)
        t = to_tensor(pil).unsqueeze(0).to('cuda')
        with torch.no_grad():
            out = model.enhance(lq=t, prompt=cap(pil), scale_by='factor', upscale=1,
                                patch_size=512, stride=512, return_type='pil')[0]
        out = np.asarray(out.convert('RGB'), dtype=np.uint8)
        if out.shape != lq.shape:
            out = np.asarray(Image.fromarray(out).resize((lq.shape[1], lq.shape[0]), Image.BICUBIC), dtype=np.uint8)
        Image.fromarray(lq).save(OUT / 'lq' / f'{i:06d}.jpg', quality=95)
        Image.fromarray(out).save(OUT / 'tgt' / f'{i:06d}.jpg', quality=95)
        i += 1
        if i % 100 == 0:
            el = time.time() - t0
            print(f'  {i}/{N}  {(el/max(i-start,1)):.2f}s/对  ETA {(N-i)*el/max(i-start,1)/60:.0f} min', flush=True)
    print(f'[done] {len(list((OUT/"lq").glob("*.jpg")))} 对 -> {OUT}', flush=True)
    print('GEN_HYPIR_DONE')


if __name__ == '__main__':
    main()
