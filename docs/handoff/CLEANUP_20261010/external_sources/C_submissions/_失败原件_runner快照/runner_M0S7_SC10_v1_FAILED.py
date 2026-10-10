"""合规 model_dir runner v14b (2026-09-08 自包含版, 规则三合规)
模型主体: VOSR 1.4B os + m0s1-s4000 LoRA(sc0.93, 已合并进权重) 完整推理链
  qwen VAE encode(mode) -> (z-mean)/std 归一化 -> dinov2l 特征 -> LightningDiT 一步 -> 反归一化 decode
完全自包含: 仅依赖 torch/torchvision/numpy; qwen VAE 为自包含裁剪版 (qwen_vae_lite)
接口: Runner(model_dir).infer(image_tensor, prompt="") -> [1,3,H,W] in [-1,1]
说明: 512x512 完整前向 (fp16), 无任何捷径/跳过; 与出图(output_dir)同一模型主体同一套权重。
"""
from __future__ import annotations
import os
import sys

# 无 triton 环境 (Windows/部分 Linux) 下 dynamo 自动编译会崩: 静默回退 eager
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')

import numpy as np  # noqa: F401  (vision_transformer 顶层需要)
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from models.lightningdit import LightningDiT  # noqa: E402
from models.qwen_vae_lite import AutoencoderKLQwenImage2D  # noqa: E402
from dinov2.models import vision_transformer as _dvt  # noqa: E402

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1)

_VAE_KEYS = ('base_dim', 'z_dim', 'dim_mult', 'num_res_blocks', 'attn_scales',
             'temperal_downsample', 'dropout', 'latents_mean', 'latents_std',
             'non_linearity')


def _build_venc(pkg, device):
    """dinov2 vitl14 (与官方 hub 构造一致: init_values=1.0, block_chunks=0, img_size=518)"""
    venc = _dvt.vit_large(patch_size=14, img_size=518, init_values=1.0,
                          block_chunks=0, ffn_layer='mlp')
    venc.load_state_dict(pkg['venc'], strict=False)
    venc.eval()
    return venc


def _build_vae(pkg, device):
    kw = {k: pkg['vae_cfg'][k] for k in _VAE_KEYS if k in pkg['vae_cfg']}
    vae = AutoencoderKLQwenImage2D(**kw)
    vae.load_state_dict(pkg['vae'], strict=True)
    vae.eval()
    return vae


class _FullChain(nn.Module):
    """1.4B 完整推理链 (fp16 + autocast, 与训练/出图一致)"""

    def __init__(self, pkg):
        super().__init__()
        self.dinov2_size = int(pkg['dinov2_size'])          # 448
        self.layer_list = list(pkg['layer_dinov2b_list'])   # [17]
        self.z_dim = int(pkg.get('z_dim', 16))
        lm = torch.tensor(pkg['vae_cfg']['latents_mean'], dtype=torch.float16).view(1, -1, 1, 1)
        ls = torch.tensor(pkg['vae_cfg']['latents_std'], dtype=torch.float16).view(1, -1, 1, 1)
        self.register_buffer('lat_mean', lm)
        self.register_buffer('lat_std_inv', 1.0 / ls)
        self.venc = _build_venc(pkg, None)
        self.vae = _build_vae(pkg, None)
        self.diT = LightningDiT(**pkg['diT_cfg'])
        self.diT.load_state_dict(pkg['diT'], strict=True)
        self.diT.forward = self.diT.forward_flexible
        self.register_buffer('mean', _MEAN)
        self.register_buffer('std', _STD)

    def forward(self, x, z):
        with torch.autocast('cuda', dtype=torch.float16):
            # dinov2l 特征 (只跑到所需层 17, 与全跑输出一致)
            raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=self.dinov2_size,
                                mode='bicubic').clamp(0., 255.)
            xn = (raw / 255.0 - self.mean) / self.std
            v = self.venc.prepare_tokens_with_masks(xn, None)
            zfeat = None
            for i, blk in enumerate(self.venc.blocks):
                v = blk(v)
                if i in self.layer_list:
                    zfeat = v[:, 1:].detach()
            # qwen VAE encode (确定性 mode) -> 归一化潜空间
            lat = self.vae.encode(x).mode()
            lat = (lat - self.lat_mean) * self.lat_std_inv
            # 一步 denoise
            u = self.diT(torch.cat([lat, z], 1),
                         torch.ones(x.shape[0], device=x.device),
                         torch.zeros(x.shape[0], device=x.device),
                         [zfeat])
            z1 = z - u
            # decode: 反归一化 -> qwen VAE decode
            sr = self.vae.decode(z1 / self.lat_std_inv + self.lat_mean)
        return sr


class Runner:
    """官方 harness 统一接口"""

    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),
                         map_location='cpu', weights_only=False)
        self.model = _FullChain(pkg).to(self.device).half().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def infer(self, image_tensor, prompt=""):
        """image_tensor: [1,3,H,W] float in [-1,1]; 返回同尺寸 [-1,1] (fp32)"""
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, _, h, w = x.shape
        # 防御: pad 到 8 的倍数 (官方测速 512x512, 无需 pad; 其余尺寸安全处理)
        ph = (-h) % 8
        pw = (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        z = torch.randn(b, self.model.z_dim, (h + ph) // 8, (w + pw) // 8,
                        device=self.device, dtype=torch.float16)
        with torch.no_grad():
            sr = self.model(x, z)
        if ph or pw:
            sr = sr[:, :, :h, :w]
        return sr.float().clamp(-1.0, 1.0)
