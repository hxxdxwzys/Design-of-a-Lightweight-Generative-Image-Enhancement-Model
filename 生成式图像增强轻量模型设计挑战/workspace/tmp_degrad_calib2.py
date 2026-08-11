"""校准 v2: 网格搜索退化管线(顺序 x blur x noise x jpeg) 拟合官方 LQ。

在 2 个 768x768 crop 上对 3 种操作顺序做粗网格, 取拟合 PSNR 最高的
(order, blur_sigma, noise_sigma, jpeg_q) 作为官方退化估计。
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
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(float(sigma))), dtype=np.float32)


def jpeg(arr, q):
    im = Image.fromarray(arr.astype(np.uint8))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)


def add_noise(arr, sigma):
    if sigma <= 0:
        return arr
    return np.clip(arr + np.random.RandomState(0).randn(*arr.shape) * sigma, 0, 255)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def pipeline(g, order, s, n, q):
    if order == "b_n_j":
        return jpeg(add_noise(blur(g, s), n), q)
    if order == "b_j_n":
        return add_noise(jpeg(blur(g, s), q), n)
    if order == "j_b_n":
        return add_noise(blur(jpeg(g, q), s), n)
    raise ValueError(order)


def estimate(case):
    gt_p = VAL / f"case{case}_gt.png"
    if not gt_p.exists():
        gt_p = VAL / f"case{case}_gt.jpg"
    gt = load(gt_p)
    lq = load(VAL / f"case{case}_lq.jpg")
    h, w = gt.shape[:2]
    crops = [(h // 2 - 384, w // 2 - 384), (h // 4 - 192, w // 4 - 192)]
    best = (None, -1)
    for y, x in crops:
        g = gt[y:y + 768, x:x + 768]
        l = lq[y:y + 768, x:x + 768]
        for order in ("b_n_j", "b_j_n", "j_b_n"):
            for s in np.arange(0.2, 4.01, 0.6):
                for q in (20, 30, 40, 55, 70, 85, 95):
                    for n in (0.0, 0.5, 1.0, 1.5):
                        p = psnr(pipeline(g, order, float(s), n, q), l)
                        if p > best[1]:
                            best = (order, p, float(s), n, q)
    return best


if __name__ == "__main__":
    for case in (1, 2, 3):
        r = estimate(case)
        print(f"case{case}: best={r}")
