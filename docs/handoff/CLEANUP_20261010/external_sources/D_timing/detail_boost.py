# -*- coding: utf-8 -*-
"""确定性细节增强算子 (2026-09-13) —— 离线出图与 runner 共用同一实现

out = x + amount * (x - blur_sigma(x)),  反射填充, 可分离高斯, 无阈值
离线(4K)与 runner(512) 使用完全相同的核与参数 => "同一模型主体 + 同一后处理"。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

SIGMA = 1.0
AMOUNT = 0.6


def gauss_kernel(sigma: float, device, dtype):
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k.to(dtype=dtype)


def blur(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kw = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, kw, groups=c)
    return x


def detail_boost(x: torch.Tensor, sigma: float = SIGMA, amount: float = AMOUNT) -> torch.Tensor:
    """x: [B,C,H,W] in [-1,1] (任意 dtype/设备); 返回同形, clamp 到 [-1,1]"""
    k = gauss_kernel(sigma, x.device, x.dtype)
    return (x + amount * (x - blur(x, k))).clamp(-1.0, 1.0)
