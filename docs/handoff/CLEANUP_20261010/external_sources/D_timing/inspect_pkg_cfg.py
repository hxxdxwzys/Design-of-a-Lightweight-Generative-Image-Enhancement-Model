# -*- coding: utf-8 -*-
"""检查 M24 权重包里的 VAE 配置 (force_upcast / 注意力) 与各模块参数量"""
import torch, json
p = r'D:/models/final_compliant/model_dir/your_model.pt'
pkg = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
print('keys:', [k for k in pkg.keys()])
print('vae_cfg:', json.dumps({k: v for k, v in pkg['vae_cfg'].items()}, ensure_ascii=False))
print('dec_cfg:', json.dumps({k: v for k, v in pkg['dec_cfg'].items()}, ensure_ascii=False, default=str)[:600])
print('diT_cfg:', json.dumps({k: v for k, v in pkg['diT_cfg'].items()}, ensure_ascii=False, default=str)[:900])
print('scale_factor:', pkg.get('scale_factor'), 'dinov2_size:', pkg.get('dinov2_size'), 'layers:', pkg.get('layer_dinov2b_list'))
for k in ('vae','diT','dec','venc'):
    if k in pkg:
        try:
            n = sum(v.numel() for v in pkg[k].values())
            print(f'  {k}: {n/1e6:.2f}M params')
        except Exception as e:
            print(f'  {k}: {e}')
