from __future__ import annotations
"""timm shim: 仅提供 LightningDiT 需要的 PatchEmbed / Mlp (官方环境无 timm)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbed(nn.Module):
    """timm PatchEmbed (vision transformer): conv patch 化 + flatten, 支持任意 img_size"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True, strict_img_size=True, dynamic_img_pad=False):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """timm Mlp"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
