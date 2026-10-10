"""同分辨率真实退化管线：HQ -> LQ (模糊 + 噪声 + JPEG，保持分辨率)

模拟赛题 LQ 退化(重JPEG压缩+噪声+模糊)。训练时对 HQ patch 在线生成 LQ。
参考 Real-ESRGAN 的退化思想，但因赛题为同分辨率复原，不含下采样。
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gaussian_noise(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
    return np.clip(arr + noise, 0, 255)


def _gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade(img: Image.Image,
            jpeg_range=(20, 95),
            noise_sigma_range=(1, 20),
            blur_radius_range=(0.0, 2.0),
            second_stage_prob=0.5) -> Image.Image:
    """HQ PIL RGB -> LQ PIL RGB (同尺寸)。"""
    # Stage 1
    r = random.uniform(*blur_radius_range)
    img = _gaussian_blur(img, r)

    arr = np.asarray(img, dtype=np.float32)
    arr = _gaussian_noise(arr, random.uniform(*noise_sigma_range))
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    q = random.randint(*jpeg_range)
    img = _jpeg(img, q)

    # Stage 2 (更轻)
    if random.random() < second_stage_prob:
        img = _gaussian_blur(img, random.uniform(0.0, 0.6))
        arr = np.asarray(img, dtype=np.float32)
        arr = _gaussian_noise(arr, random.uniform(0, 5))
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        img = _jpeg(img, random.randint(60, 95))

    return img
