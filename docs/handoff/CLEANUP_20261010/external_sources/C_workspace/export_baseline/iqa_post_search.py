# -*- coding: utf-8 -*-
"""OPT-04 Phase A：官方风格 NR-IQA 代理对「可解释后处理」的响应测试（CPU 图像处理 + GPU 指标推理）。

用法: python iqa_post_search.py <img_dir> [--n 6] [--musiq-n 2]
输出: D:/models/opt04/evidence/iqa_post_search.json
说明: 只做本地测量，不修改交付件；后处理若最终采用，必须实现在 runner 内（保证重跑可复现）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def variants():
    def v_identity(im):
        return im

    def v_unsharp_mild(im):
        return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    def v_unsharp_strong(im):
        return im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=120, threshold=2))

    def v_contrast_105(im):
        return ImageEnhance.Contrast(im).enhance(1.05)

    def v_saturation_105(im):
        return ImageEnhance.Color(im).enhance(1.05)

    def v_clahe(im):
        import cv2
        arr = np.asarray(im)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    return [('v0_identity', v_identity), ('v1_unsharp_mild', v_unsharp_mild),
            ('v2_unsharp_strong', v_unsharp_strong), ('v3_contrast105', v_contrast_105),
            ('v4_saturation105', v_saturation_105), ('v5_clahe15', v_clahe)]


def main():
    d = Path(sys.argv[1])
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 6
    musiq_n = int(sys.argv[sys.argv.index('--musiq-n') + 1]) if '--musiq-n' in sys.argv else 2
    imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])[:n]
    import torch
    import pyiqa
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = {k: pyiqa.create_metric(k, device=dev) for k in ('clipiqa', 'maniqa', 'niqe')}
    if musiq_n > 0:
        metrics['musiq'] = pyiqa.create_metric('musiq', device=dev)
    out = {'schema': 'OPT04-post-search/1', 'dir': str(d), 'n_images': len(imgs),
           'images': [p.name for p in imgs], 'device': str(dev), 'rows': {}}
    tmp = Path(r'D:\models\opt04\tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    for name, fn in variants():
        scores = {'clipiqa': [], 'maniqa': [], 'niqe': [], 'musiq': []}
        t0 = time.perf_counter()
        for i, p in enumerate(imgs):
            im = Image.open(p).convert('RGB')
            out_im = fn(im)
            if name == 'v0_identity':
                q = p
            else:
                q = tmp / ('%s_%s' % (name, p.name))
                out_im.save(q, format='JPEG', quality=98, subsampling=0, optimize=False,
                            progressive=False)
            with torch.no_grad():
                for k, m in metrics.items():
                    if k == 'musiq' and (musiq_n <= 0 or i >= musiq_n):
                        continue
                    scores[k].append(float(m(str(q)).item()))
        out['rows'][name] = {
            'mean': {k: (float(np.mean(v)) if v else None) for k, v in scores.items()},
            'n': {k: len(v) for k, v in scores.items()},
            'wall_s': round(time.perf_counter() - t0, 1)}
        m = out['rows'][name]['mean']
        print('%-18s clipiqa=%.4f maniqa=%.4f niqe=%.4f musiq=%s (%.0fs)'
              % (name, m['clipiqa'], m['maniqa'], m['niqe'],
                 ('%.2f' % m['musiq']) if m['musiq'] else 'n/a',
                 out['rows'][name]['wall_s']), flush=True)
    p = Path(r'D:\models\opt04\evidence')
    p.mkdir(parents=True, exist_ok=True)
    (p / 'iqa_post_search.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                            encoding='utf-8')
    print('written', p / 'iqa_post_search.json')


if __name__ == '__main__':
    main()
