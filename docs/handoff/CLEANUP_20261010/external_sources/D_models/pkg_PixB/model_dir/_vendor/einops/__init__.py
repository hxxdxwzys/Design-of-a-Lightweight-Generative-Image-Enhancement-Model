from __future__ import annotations
"""einops shim (2026-09-14 扩展): 支持 PixRestore + 本项目用到的模式"""
import torch


def rearrange(tensor, pattern, **axes_lengths):
    if pattern == '... (d r) -> ... d r':
        r = axes_lengths['r']
        return tensor.view(*tensor.shape[:-1], -1, r)
    if pattern == '... d r -> ... (d r)':
        return tensor.flatten(-2)
    if pattern == 'b -> b 1 1 1':
        b = tensor.shape[0]
        return tensor.reshape(b, 1, 1, 1)
    if pattern == ' -> 1 1 1 1':
        return tensor.reshape(1, 1, 1, 1)
    if pattern == 'b c h w -> b (h w) c':
        b, c, h, w = tensor.shape
        return tensor.reshape(b, c, h * w).transpose(1, 2)
    if pattern == 'b (h w) c -> b c h w':
        b, n, c = tensor.shape
        import math
        h = int(math.sqrt(n))
        return tensor.transpose(1, 2).reshape(b, c, h, n // h)
    raise NotImplementedError(f'einops pattern not supported: {pattern}')


def repeat(tensor, pattern, **axes_lengths):
    if pattern == '... n -> ... (n r)':
        r = axes_lengths['r']
        return tensor.repeat_interleave(r, dim=-1)
    if pattern == 'h n -> h (n r)':
        r = axes_lengths['r']
        return tensor.repeat_interleave(r, dim=-1)
    if pattern == 'w n -> w (n r)':
        r = axes_lengths['r']
        return tensor.repeat_interleave(r, dim=-1)
    raise NotImplementedError(f'einops pattern not supported: {pattern}')
