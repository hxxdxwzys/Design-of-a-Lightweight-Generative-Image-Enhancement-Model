# -*- coding: utf-8 -*-
"""DELIV-02 终端 8 位恢复（唯一 PNG/JPEG 编码入口）。

wavelet 已经在内部按原 floor 算法量化为整数色阶并以 FP32 [-1,1] 返回；这里只**恢复**
已量化的色阶，不改变模型数值图，也禁止再次 floor/truncate（旧实现的二次截断会造成
大范围 1-2 色阶差异，见 review DEL-R01–R05）。

规则（§0F.4）：u8 = round((clamp(y,-1,1)+1)*127.5)，在 CPU float64 上计算；
先检查 finite/范围，再检查到整数网格的残差 ≤1e-4 色阶；超出即拒绝（不能用 round 掩盖任意连续输出）。
"""
import numpy as np

GRID_TOL_LEVELS = 1e-4


class PixelIOError(RuntimeError):
    pass


def to_u8(y):
    """[-1,1] FP32/FP64 -> uint8 HxWxC（最近色阶恢复）。

    y: [1,3,H,W] torch.Tensor 或 np.ndarray -> 返回 np.uint8 (H,W,C)。
    """
    arr = _as_numpy(y)
    if not np.isfinite(arr).all():
        raise PixelIOError("输出含 NaN/Inf，拒绝编码")
    if arr.min() < -1.0 - 1e-6 or arr.max() > 1.0 + 1e-6:
        raise PixelIOError("输出超出 [-1,1] 范围：%.6f..%.6f" % (arr.min(), arr.max()))
    levels = (np.clip(arr.astype(np.float64), -1.0, 1.0) + 1.0) * 127.5
    residual = np.abs(levels - np.round(levels))
    worst = float(residual.max())
    if worst > GRID_TOL_LEVELS:
        raise PixelIOError("到整数网格残差 %.3e 色阶 > %.0e：输出不是已量化的色阶，禁止用 round 掩盖"
                           % (worst, GRID_TOL_LEVELS))
    u8 = np.round(levels).astype(np.uint8)
    return np.transpose(u8, (1, 2, 0)) if u8.shape[0] == 3 else u8


def _as_numpy(y):
    if hasattr(y, "detach"):
        y = y.detach().cpu().float().numpy()
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise PixelIOError("B 必须为 1")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise PixelIOError("期望 [1,3,H,W] 或 [3,H,W]，实为 %s" % (arr.shape,))
    return arr


def grid_residual(y):
    """返回 (最大残差色阶, finite, min, max)，供证据记录。"""
    arr = _as_numpy(y)
    finite = bool(np.isfinite(arr).all())
    if not finite:
        return {"max_residual_levels": None, "finite": False, "min": None, "max": None}
    levels = (np.clip(arr.astype(np.float64), -1.0, 1.0) + 1.0) * 127.5
    return {"max_residual_levels": float(np.abs(levels - np.round(levels)).max()),
            "finite": True, "min": float(arr.min()), "max": float(arr.max())}


def u8_delta(a_u8, b_u8):
    """比较两份 uint8 RGB：返回每通道最大差、任一通道不同的像素数与占比。"""
    a = np.asarray(a_u8).astype(np.int16)
    b = np.asarray(b_u8).astype(np.int16)
    if a.shape != b.shape:
        raise PixelIOError("形状不一致：%s vs %s" % (a.shape, b.shape))
    d = np.abs(a - b)
    h, w = d.shape[0], d.shape[1]
    any_diff = (d.max(axis=2) > 0)
    return {"max_channel_diff": int(d.max()), "n_pixels_any_channel_diff": int(any_diff.sum()),
            "frac_pixels_any_channel_diff": float(any_diff.sum()) / float(h * w),
            "identical": bool(d.max() == 0), "h": int(h), "w": int(w)}
