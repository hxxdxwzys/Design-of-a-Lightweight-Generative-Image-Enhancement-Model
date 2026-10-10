# -*- coding: utf-8 -*-
"""蒸馏数据生成 v2 (批量版): (退化LQ, M24输出) 成对 patch -> D:/datasets/distill256/{lq,tgt}
教师(M24)支持 batch -> 一次算 4 张, 省掉逐张的固定开销 (0.65s/对 -> ~0.2s/对)
"""
from __future__ import annotations
import io, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import importlib.util
MD = r'D:/models/final_compliant/model_dir'
sys.path.insert(0, MD)
spec = importlib.util.spec_from_file_location('m24runner', os.path.join(MD, 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PATCH = sorted(Path(r'D:/datasets/tiny_patches').glob('*.jpg'))
OUT = Path(r'D:/datasets/distill256')
(OUT / 'lq').mkdir(parents=True, exist_ok=True)
(OUT / 'tgt').mkdir(parents=True, exist_ok=True)
N = 3000
B = 4
POOL = ThreadPoolExecutor(max_workers=10)


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
    t0 = time.time()
    start = len(list((OUT / 'lq').glob('*.jpg')))
    print(f'[load] M24 teacher ... (已有 {start} 对)', flush=True)
    R = mod.Runner(MD)
    print(f'[teacher] ready {time.time()-t0:.0f}s | 目标 {N} 对, batch {B}', flush=True)
    small = lambda a: np.asarray(Image.fromarray(a).resize((256, 256), Image.BICUBIC), dtype=np.uint8)
    i = start
    while i < N:
        lqs = list(POOL.map(load_lq, range(B)))
        t = torch.from_numpy(np.stack(lqs)).permute(0, 3, 1, 2).to('cuda').float().div_(127.5).sub_(1)
        with torch.no_grad():
            y = R.infer(t)
        for j in range(B):
            if i + j >= N:
                break
            tgt = np.clip((y[j].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
            # 存 512 域原图 -> 训练时按 down 自行降采样 (一个数据集两用: down=1 与 down=2)
            Image.fromarray(lqs[j]).save(OUT / 'lq' / f'{i+j:06d}.jpg', quality=95)
            Image.fromarray(tgt).save(OUT / 'tgt' / f'{i+j:06d}.jpg', quality=95)
        i += B
        if i % 200 == 0:
            el = time.time() - t0
            print(f'  {i}/{N}  {(el/(i-start)):.2f}s/对  ETA {(N-i)*el/max(i-start,1)/60:.0f} min', flush=True)
    print(f'[done] -> {OUT}  共 {len(list((OUT/"lq").glob("*.jpg")))} 对, {(time.time()-t0)/60:.1f} min', flush=True)
    print('GEN_DISTILL_DONE')


if __name__ == '__main__':
    main()
