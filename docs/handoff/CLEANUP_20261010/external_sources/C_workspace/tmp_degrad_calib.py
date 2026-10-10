"""校准官方 LQ 退化参数: 从 3 对 (GT, LQ) 估计 blur σ / noise σ / JPEG q。

对每个 case 取 3 个 768x768 固定位置 crop:
  - blur: 对 GT 高斯模糊 σ∈[0,2.4] 网格, 取 PSNR(LQ) 最大的 σ
  - noise: 最优 blur 后的残差, 在低梯度(平坦)区域估稳健 std
  - jpeg: 对 blur 后参考图按 q∈[10..100] 重编码, 取 MSE(LQ) 最小的 q
输出逐 case 估计 + 建议的 degradation.py 参数范围。
"""
import io
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def blur(arr, sigma):
    if sigma <= 0:
        return arr
    im = Image.fromarray(arr.astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(sigma))), dtype=np.float32)


def jpeg(arr, q):
    im = Image.fromarray(arr.astype(np.uint8))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def estimate(case):
    gt_p = VAL / f"case{case}_gt.png"
    if not gt_p.exists():
        gt_p = VAL / f"case{case}_gt.jpg"
    gt = load(gt_p)
    lq = load(VAL / f"case{case}_lq.jpg")
    h, w = gt.shape[:2]
    crops = [(h // 2 - 384, w // 2 - 384), (h // 4 - 192, w // 4 - 192),
             (3 * h // 4 - 192, 3 * w // 4 - 192)]
    out = []
    for y, x in crops:
        g = gt[y:y + 768, x:x + 768]
        l = lq[y:y + 768, x:x + 768]
        sigs = np.arange(0.0, 2.41, 0.2)
        ps = [psnr(blur(g, s), l) for s in sigs]
        s_best = float(sigs[int(np.argmax(ps))])
        gb = blur(g, s_best)
        # flat-area robust noise sigma
        res = l - gb
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(res, (3, 3, 1)).squeeze(-1)
        local_std = sw.std(axis=(2, 3, 4))
        flat = np.sort(local_std.ravel())[: max(1, len(local_std.ravel()) // 10)]
        n_best = float(np.median(flat))
        # jpeg q grid on blur-only reference
        qs = list(range(10, 101, 10))
        m = [float(np.mean((jpeg(gb, q) - l) ** 2)) for q in qs]
        q_best = int(qs[int(np.argmin(m))])
        out.append((round(s_best, 2), round(n_best, 2), q_best, round(float(ps[int(np.argmax(ps))]), 2)))
    return out


if __name__ == "__main__":
    for case in (1, 2, 3):
        print(f"case{case}: (blur_sigma, noise_sigma, jpeg_q, best_fit_psnr) per crop:")
        for r in estimate(case):
            print("   ", r)
