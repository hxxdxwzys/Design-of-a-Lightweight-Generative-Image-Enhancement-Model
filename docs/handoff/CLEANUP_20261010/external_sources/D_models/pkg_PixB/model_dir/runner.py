# -*- coding: utf-8 -*-
"""PixRestore 合规 model_dir runner (2026-09-14 自包含版)

模型主体: PixRestore (像素空间条件流匹配扩散 Transformer, PolyU+OPPO 2026.8)
  完整推理链: DINOv2 多层特征 -> 层路由条件 -> 1 步流匹配采样 (pixel DiT)
  权重: pixrestore-b (DiT 124M + DINOv2-B 86M), fp16
接口: Runner(model_dir).infer(image_tensor, prompt="") -> [1,3,H,W] in [-1,1]
说明: 512x512 完整前向 (1 步), 无任何捷径/跳过; 与出图(output_dir)同一模型主体同一套权重。
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_MODEL_DIR, '_vendor')
_PATH_BEFORE = list(sys.path)
if os.path.isdir(_VENDOR_DIR):
    sys.path.insert(0, _VENDOR_DIR)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion  # noqa: E402
from dinov2.models import vision_transformer as _dvt  # noqa: E402

# ── 交还 import 环境 (官方 IQA 评测器与我们在同一进程, 不能遮蔽 timm/einops) ──
_MD_ABS = os.path.abspath(_MODEL_DIR)
_VD_ABS = os.path.abspath(_VENDOR_DIR)
sys.path[:] = [p for p in _PATH_BEFORE if os.path.abspath(p) not in (_MD_ABS, _VD_ABS)]
for _n in [n for n in sys.modules if n.split('.')[0] in ('timm', 'einops', 'diffusers', 'fairscale')]:
    _f = getattr(sys.modules[_n], '__file__', None) or ''
    _fa = os.path.abspath(_f) if _f else ''
    if _fa.startswith(_MD_ABS + os.sep) or _fa.startswith(_VD_ABS + os.sep):
        del sys.modules[_n]
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.append(_VENDOR_DIR)
if _MODEL_DIR not in sys.path:
    sys.path.append(_MODEL_DIR)

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1)


def _build_dit(cfg):
    return LightningDiT_PixelDiffusion(
        input_size=cfg['resolution'], patch_size=cfg['patch_size'], in_channels=6, out_channels=3,
        hidden_size=cfg['hidden_size'], depth=cfg['depth'], num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'], z_dims=cfg['encoder_dim'] if cfg.get('use_venc', True) else None,
        encdim_ratio=cfg['encoder_dim_ratio'], use_qknorm=cfg['use_qknorm'], use_swiglu=cfg['use_swiglu'],
        use_rope=cfg['use_rope'], use_rmsnorm=cfg['use_rmsnorm'], adain_single=True,
        num_fused_layers=len(cfg['encoder_layers']), pca_dim=cfg['bottleneck_dim'],
        use_bottleneck_patch_embed=True, use_dino_layer_router=cfg['use_dino_layer_router'],
        dino_layer_router_mode=cfg['dino_layer_router_mode'],
        same_condition_layer_weights=cfg['same_condition_layer_weights'],
        gate_temperature=cfg['gate_temperature'], feature_norm=cfg['feature_norm'],
        use_lq_degradation_token=bool(cfg.get('use_lq_degradation_token', False)),
        register_local_gate_logit_scale=bool(cfg.get('register_local_gate_logit_scale', False)))


def _build_venc():
    return _dvt.vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, ffn_layer='mlp')


def _forward_with_features(self, image, masks=None):
    features = {}
    tokens = self.prepare_tokens_with_masks(image, masks)
    for index, block in enumerate(self.blocks):
        tokens = block(tokens)
        features[index] = tokens[:, 1:]
    return features, self.norm(tokens)[:, 1:]


class Runner:
    """官方 harness 统一接口"""

    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu',
                             weights_only=False, mmap=True)
        except TypeError:
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')
        cfg = pkg['cfg']
        self.cfg = cfg
        self.layers = list(cfg['encoder_layers'])
        self.enc_size = int(cfg['encoder_input_size'])

        model = _build_dit(cfg)
        model.load_state_dict(pkg['dit'], strict=False)
        self.model = model.to(self.device).half().eval()

        venc = _build_venc()
        venc.load_state_dict(pkg['venc'], strict=False)
        venc.forward_with_features = _forward_with_features.__get__(venc, type(venc))
        self.venc = venc.to(self.device).half().eval()
        self.venc.requires_grad_(False)

        from types import SimpleNamespace
        self.flow = PixelDiffusion(
            flow_ratio=cfg['flow_ratio'], time_dist=cfg['time_distribution'], alpha=cfg['alpha'],
            z_start='noise', cfg_ratio=cfg['cfg_ratio'], cfg_scale=cfg['cfg_scale'],
            image_size=cfg['resolution'], channels=3, interp_type='lin', uncond_type='zero',
            norm_p=cfg['adaptive_loss_power'], accelerator=None, t_start=0, t_end=1, use_cos=False,
            args=SimpleNamespace(**cfg))
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _features(self, x):
        # 与 pixrestore.vision.preprocess_dinov2 完全一致: [-1,1] -> [0,1] -> 448 -> 归一化
        raw = F.interpolate(x.float().add(1).mul(0.5), size=(self.enc_size, self.enc_size),
                            mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
        xn = ((raw - _MEAN.float().to(x.device)) / _STD.float().to(x.device)).half()
        feats, final = self.venc.forward_with_features(xn)
        if feats:
            feats[max(feats)] = final
        return [feats[i] for i in self.layers]

    @torch.no_grad()
    def infer(self, image_tensor, prompt=""):
        """image_tensor: [1,3,H,W] float in [-1,1]; 返回同尺寸 [-1,1] (fp32)"""
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, c, h, w = x.shape
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        fea = self._features(x)
        with torch.autocast('cuda', dtype=torch.float16):
            out = self.flow.sample_multistep_fm(self.model, x, venc_fea=fea, n_steps=1, schedule='linear')
        out = out.float()
        if ph or pw:
            out = out[:, :, :h, :w]
        return out.clamp(-1.0, 1.0)
