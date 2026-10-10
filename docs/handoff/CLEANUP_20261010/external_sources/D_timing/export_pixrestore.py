# -*- coding: utf-8 -*-
"""PixRestore 合规导出: config + DiT + DINOv2 权重 -> your_model.pt (fp16)"""
import json, sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch

MD = Path(r'D:/models/pkg_PixB/model_dir')
SIZE = 'b'
CKPT = Path(rf'D:/models/newsrc/PixRestore/pixrestore-{SIZE}')
sys.path.insert(0, str(MD))
sys.path.insert(0, r'D:/models/newsrc/PixRestore_code')

from safetensors.torch import load_file
cfg = json.loads((CKPT / 'config.json').read_text(encoding='utf-8'))
# 只保留构造与推理需要的键
KEEP = ('resolution', 'patch_size', 'bottleneck_dim', 'hidden_size', 'depth', 'num_heads', 'mlp_ratio',
        'encoder_dim_ratio', 'use_qknorm', 'use_swiglu', 'use_rope', 'use_rmsnorm', 'encoder_type',
        'encoder_dim', 'encoder_input_size', 'encoder_layers', 'feature_norm', 'gate_temperature',
        'dino_layer_router_mode', 'use_dino_layer_router', 'same_condition_layer_weights',
        'use_lq_degradation_token', 'flow_ratio', 'time_distribution', 'alpha', 'cfg_ratio', 'cfg_scale',
        'adaptive_loss_power', 'use_venc', 'cond_strength_aelq_test', 'cond_strength_venc')
# 保留完整 config (PixelDiffusion 会读 aug_noise 等键) + 确保 KEEP 里的键都在
cfg = dict(cfg)
for k in KEEP:
    if k not in cfg:
        raise SystemExit(f'config 缺键: {k}')
dit = load_file(str(CKPT / 'clean_weights' / 'ema_model.safetensors'))
print(f'[dit] {len(dit)} tensors', flush=True)
from dinov2.models import vision_transformer as _dvt
venc = _dvt.vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, ffn_layer='mlp')
sd = torch.load(r'D:/models/newsrc/dinov2/dinov2_vitb14_pretrain.pth', map_location='cpu')
msg = venc.load_state_dict(sd, strict=False)
print('[venc] missing', len(msg.missing_keys), 'unexpected', len(msg.unexpected_keys), flush=True)
pkg = {'format': 'pixrestore_compliant_v1', 'size': SIZE, 'cfg': cfg,
       'dit': {k: v.half() for k, v in dit.items()},
       'venc': {k: v.half() for k, v in venc.state_dict().items()}}
torch.save(pkg, MD / 'your_model.pt')
n = sum(v.numel() for v in pkg['dit'].values()) + sum(v.numel() for v in pkg['venc'].values())
print(f'[save] {MD/"your_model.pt"} {(MD/"your_model.pt").stat().st_size/1e6:.0f}MB | {n/1e6:.1f}M params', flush=True)
print('EXPORT_PIX_DONE')
