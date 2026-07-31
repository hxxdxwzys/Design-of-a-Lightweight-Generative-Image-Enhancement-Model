"""PatchGAN 判别器 (谱归一化 + InstanceNorm, 返回 logit 与中间特征用于特征匹配)"""
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._block(in_ch, base, norm=False),
            self._block(base, base * 2),
            self._block(base * 2, base * 4),
            self._block(base * 4, base * 8, stride=1),
        ])
        self.final = spectral_norm(nn.Conv2d(base * 8, 1, 4, 1, 1))

    def _block(self, i, o, stride=2, norm=True):
        layers = [spectral_norm(nn.Conv2d(i, o, 4, stride, 1, padding_mode="reflect"))]
        if norm:
            layers.append(nn.InstanceNorm2d(o, affine=True))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        logit = self.final(x)
        return logit, feats
