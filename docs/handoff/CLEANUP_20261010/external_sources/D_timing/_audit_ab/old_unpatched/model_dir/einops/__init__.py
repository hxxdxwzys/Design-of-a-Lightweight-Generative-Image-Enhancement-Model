from __future__ import annotations
"""einops shim: 仅支持本项目用到的 3 种简单模式 (rearrange/repeat)
- rearrange(x, '... (d r) -> ... d r', r=2): 最后维拆两半
- rearrange(x, '... d r -> ... (d r)'): 最后两维合并
- repeat(x, '... n -> ... (n r)', r=2): 最后维逐元素复制
"""
import torch


def rearrange(tensor, pattern, **axes_lengths):
    if pattern == '... (d r) -> ... d r':
        r = axes_lengths['r']
        return tensor.view(*tensor.shape[:-1], -1, r)
    if pattern == '... d r -> ... (d r)':
        return tensor.flatten(-2)
    raise NotImplementedError(f'einops pattern not supported: {pattern}')


def repeat(tensor, pattern, **axes_lengths):
    if pattern == '... n -> ... (n r)':
        r = axes_lengths['r']
        return tensor.repeat_interleave(r, dim=-1)
    raise NotImplementedError(f'einops pattern not supported: {pattern}')
