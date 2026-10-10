# -*- coding: utf-8 -*-
"""测试集 LQ 真实退化分布分析 (2026-08-14, CPU only, 不抢 GPU)

对 100 张测试 LQ + 3 张 val LQ 测:
  1) JPEG 质量因子 (量化表反推, 逐张精确)
  2) 模糊强度 (灰度 1024 宽, 高斯残差方差) -- 用与训练脚本同款合成退化标定, 映射回 sigma
  3) 噪声强度 (灰度原生分辨率, 中值残差 std) -- 同样标定映射回 n
标定: 3 张 DIV2K HQ, 按训练同款顺序 clean -> jpeg(q85) -> blur(sigma) -> noise(n)
输出: D:/tmp_timing/degradation_analysis.json
"""
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
HQ_DIR = Path(r'D:/datasets/div2k/pngtxt_ori')
OUT = Path(r'D:/tmp_timing/degradation_analysis.json')

STD_Q50_LUM = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def jpeg_quality(p):
    """从 JPEG 量化表反推质量因子 (1-100); 非 JPEG 或反推失败返回 None"""
    try:
        img = Image.open(p)
        qt = img.quantization
        if not qt:
            return None
        q = qt.get(0) or qt.get(1)
        if q is None:
            return None
        q = np.asarray(q, dtype=np.float64)
        if q.sum() <= 64:  # 全 1 -> 质量 100
            return 100
        best_q, best_d = None, 1e18
        for cand in range(1, 101):
            scale = 5000.0 / cand if cand < 50 else 200.0 - 2.0 * cand
            pred = np.clip(np.round(STD_Q50_LUM * scale), 1, 255)
            d = np.mean(np.abs(pred - q))
            if d < best_d:
                best_d, best_q = d, cand
        return best_q
    except Exception as e:
        return None


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def blur_metric(img):
    """灰度 1024 宽, 1.5 半径高斯残差的均方 -> 越锐利值越大"""
    g = img.convert('L')
    w = 1024
    h = max(1, int(round(g.height * w / g.width)))
    g = g.resize((w, h), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    b = np.asarray(g.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def noise_metric(img):
    """原生分辨率灰度, 3x3 中值残差 std (噪声越大值越大)"""
    g = img.convert('L')
    a = np.asarray(g, dtype=np.float32)
    m = np.asarray(g.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    return float(np.std(a - m))


def main():
    t0 = time.time()
    hq_imgs = sorted(HQ_DIR.glob('*.png'))[:3]
    print(f'[calib] HQ 标定图: {[p.name for p in hq_imgs]}', flush=True)

    # ---- 标定: blur sigma -> metric ----
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    blur_cal = {s: [] for s in sigmas}
    for p in hq_imgs:
        base = Image.open(p).convert('RGB')
        for s in sigmas:
            im = _jpeg(base, 85)
            im = im.filter(ImageFilter.GaussianBlur(radius=s))
            blur_cal[s].append(blur_metric(im))
    blur_map = {s: float(np.mean(v)) for s, v in blur_cal.items()}
    print('[calib] blur sigma -> metric:', flush=True)
    for s in sigmas:
        print(f'    sigma={s}: {blur_map[s]:.2f}', flush=True)

    # ---- 标定: noise n -> metric ----
    noises = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    noise_cal = {n: [] for n in noises}
    for p in hq_imgs:
        base = Image.open(p).convert('RGB')
        for n in noises:
            im = _jpeg(base, 85)
            im = im.filter(ImageFilter.GaussianBlur(radius=2.0))
            arr = np.asarray(im, dtype=np.float32)
            arr = np.clip(arr + np.random.RandomState(0).randn(*arr.shape).astype(np.float32) * n, 0, 255)
            noise_cal[n].append(noise_metric(Image.fromarray(arr.astype(np.uint8))))
    noise_map = {n: float(np.mean(v)) for n, v in noise_cal.items()}
    print('[calib] noise n -> metric:', flush=True)
    for n in noises:
        print(f'    n={n}: {noise_map[n]:.2f}', flush=True)

    def inv_map(x, keys, mapping, lo, hi):
        """在单调映射上线性插值反解 (keys 升序, 值单调)"""
        ks = sorted(keys)
        vs = [mapping[k] for k in ks]
        # 假设值随 key 单调; blur 值随 sigma 递减
        xs = np.array(vs)
        if xs[0] <= xs[-1]:
            xs = xs[::-1]
            ks = ks[::-1]
        if x >= xs[0]:
            return ks[0]
        if x <= xs[-1]:
            return ks[-1]
        for i in range(len(xs) - 1):
            if xs[i] >= x >= xs[i + 1]:
                frac = (xs[i] - x) / max(xs[i] - xs[i + 1], 1e-12)
                return float(ks[i] + frac * (ks[i + 1] - ks[i]))
        return float(ks[-1])

    # ---- 逐图分析 ----
    rows = []
    for tag, d in [('test', TEST_DIR), ('val', VAL_DIR)]:
        files = sorted(d.glob('*.jpg'))
        if tag == 'val':
            files = [f for f in files if '_lq' in f.name]
        for p in files:
            try:
                img = Image.open(p).convert('RGB')
                q = jpeg_quality(p)
                bm = blur_metric(img)
                nm = noise_metric(img)
                es = inv_map(bm, blur_map.keys(), blur_map, 0.3, 6.0)
                en = inv_map(nm, noise_map.keys(), noise_map, 0.0, 4.0)
                rows.append({
                    'tag': tag, 'name': p.name, 'w': img.width, 'h': img.height,
                    'size_kb': p.stat().st_size // 1024,
                    'jpeg_q': q, 'blur_metric': round(bm, 3), 'est_sigma': round(es, 3),
                    'noise_metric': round(nm, 3), 'est_noise_n': round(en, 3),
                })
            except Exception as e:
                rows.append({'tag': tag, 'name': p.name, 'error': str(e)})

    def stats(key, rws):
        vals = [r[key] for r in rws if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        a = np.array(vals)
        return {'n': len(vals), 'mean': round(float(a.mean()), 3), 'median': round(float(np.median(a)), 3),
                'p10': round(float(np.percentile(a, 10)), 3), 'p90': round(float(np.percentile(a, 90)), 3),
                'min': round(float(a.min()), 3), 'max': round(float(a.max()), 3)}

    test_rows = [r for r in rows if r['tag'] == 'test' and 'error' not in r]
    val_rows = [r for r in rows if r['tag'] == 'val' and 'error' not in r]
    summary = {
        'test': {k: stats(k, test_rows) for k in ['jpeg_q', 'est_sigma', 'est_noise_n', 'blur_metric', 'noise_metric']},
        'val': {k: stats(k, val_rows) for k in ['jpeg_q', 'est_sigma', 'est_noise_n', 'blur_metric', 'noise_metric']},
        'training_sampler_ranges': {'jpeg_q': [50, 96], 'blur_sigma': [1.2, 4.2], 'noise_n': [0.0, 1.5]},
    }
    # 训练范围外的比例
    for tag, rws in [('test', test_rows), ('val', val_rows)]:
        out_q = sum(1 for r in rws if r['jpeg_q'] is not None and not (50 <= r['jpeg_q'] <= 96)) / len(rws)
        out_s = sum(1 for r in rws if not (1.2 <= r['est_sigma'] <= 4.2)) / len(rws)
        out_n = sum(1 for r in rws if not (0.0 <= r['est_noise_n'] <= 1.5)) / len(rws)
        summary[tag]['outside_training_range_frac'] = {'jpeg_q': round(out_q, 3), 'sigma': round(out_s, 3), 'noise': round(out_n, 3)}

    out = {'per_image': rows, 'summary': summary,
           'blur_calibration': {str(k): round(v, 2) for k, v in blur_map.items()},
           'noise_calibration': {str(k): round(v, 2) for k, v in noise_map.items()},
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('\n===== 汇总 =====', flush=True)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print(f'[saved] {OUT} ({(time.time() - t0) / 60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
