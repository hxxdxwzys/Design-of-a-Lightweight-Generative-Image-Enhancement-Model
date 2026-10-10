# -*- coding: utf-8 -*-
"""PixRestore 质量/时延测试 (2026-09-14)
- val 3 张: 512 中心裁切 -> 1 步复原 -> 与 GT 比 PSNR/SSIM/LPIPS (对比 M24: 24.559/0.9658/0.4408)
- 512 时延 -> 加速比
用法: python test_pixrestore.py --size b [--steps 1]
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

REPO = Path(r'D:/models/newsrc/PixRestore_code')
sys.path.insert(0, str(REPO))
from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
from pixrestore.vision import extract_layers, load_dinov2

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='b')
ap.add_argument('--steps', type=int, default=1)
ap.add_argument('--n', type=int, default=3)
a = ap.parse_args()

CKPT = Path(rf'D:/models/newsrc/PixRestore/pixrestore-{a.size}')
cfg = json.loads((CKPT / 'config.json').read_text(encoding='utf-8'))
ENC_CKPT = {'dinov2s': 'dinov2_vits14_pretrain.pth', 'dinov2b': 'dinov2_vitb14_pretrain.pth',
            'dinov2l': 'dinov2_vitl14_pretrain.pth'}.get(cfg['encoder_type'], 'dinov2_vits14_pretrain.pth')
cfg['dinov2_repository'] = r'D:/models/newsrc/dinov2_repo'
cfg['dinov2_checkpoint'] = str(Path(r'D:/models/newsrc/dinov2') / ENC_CKPT)
config = SimpleNamespace(**cfg)
dev = 'cuda'

model = LightningDiT_PixelDiffusion(
    input_size=config.resolution, patch_size=config.patch_size, in_channels=6, out_channels=3,
    hidden_size=config.hidden_size, depth=config.depth, num_heads=config.num_heads,
    mlp_ratio=config.mlp_ratio, z_dims=config.encoder_dim if config.use_venc else None,
    encdim_ratio=config.encoder_dim_ratio, use_qknorm=config.use_qknorm, use_swiglu=config.use_swiglu,
    use_rope=config.use_rope, use_rmsnorm=config.use_rmsnorm, adain_single=True,
    num_fused_layers=len(config.encoder_layers), pca_dim=config.bottleneck_dim,
    use_bottleneck_patch_embed=True, use_dino_layer_router=config.use_dino_layer_router,
    dino_layer_router_mode=config.dino_layer_router_mode,
    same_condition_layer_weights=config.same_condition_layer_weights,
    gate_temperature=config.gate_temperature, feature_norm=config.feature_norm,
    use_lq_degradation_token=bool(getattr(config, 'use_lq_degradation_token', False)),
    register_local_gate_logit_scale=bool(getattr(config, 'register_local_gate_logit_scale', False)))
st = load_file(str(CKPT / 'clean_weights' / 'ema_model.safetensors'))
msg = model.load_state_dict(st, strict=False)
print('[weights] missing:', len(msg.missing_keys), 'unexpected:', len(msg.unexpected_keys), flush=True)
model = model.to(dev).eval().requires_grad_(False)
flow = PixelDiffusion(flow_ratio=config.flow_ratio, time_dist=config.time_distribution, alpha=config.alpha,
                      z_start='noise', cfg_ratio=config.cfg_ratio, cfg_scale=config.cfg_scale,
                      image_size=config.resolution, channels=3, interp_type='lin', uncond_type='zero',
                      norm_p=config.adaptive_loss_power, accelerator=None, t_start=0, t_end=1,
                      use_cos=False, args=config)
enc = load_dinov2(config.encoder_type, dev, repository=config.dinov2_repository,
                  checkpoint=config.dinov2_checkpoint)
nparam = sum(p.numel() for p in model.parameters()) / 1e6
print(f'[model] pixrestore-{a.size}  DiT {nparam:.1f}M params | encoder {config.encoder_type}', flush=True)

@torch.no_grad()
def restore(lq):
    fea = extract_layers(enc, lq, config.encoder_layers, config.encoder_input_size)
    with torch.autocast('cuda', dtype=torch.float16):
        out = flow.sample_multistep_fm(model, lq, venc_fea=fea, n_steps=a.steps, schedule='linear')
    return out

# 时延
x = torch.randn(1, 3, 512, 512, device=dev).clamp(-1, 1)
with torch.no_grad():
    for _ in range(3):
        restore(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        restore(x)
    torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 10 * 1000
print(f'[latency] {a.steps} 步 512: {ms:.1f} ms -> 加速比 ~{285.6/ms:.2f}x', flush=True)

import lpips as lpips_mod
lp = lpips_mod.LPIPS(net='alex', verbose=False)
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
def psnr(p, g):
    m = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    return 99.0 if m <= 0 else 10 * np.log10(255.0 ** 2 / m)
def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var()
    cov = ((p - m1) * (g - m2)).mean()
    return ((2*m1*m2 + C1) * (2*cov + C2)) / ((m1**2 + m2**2 + C1) * (v1 + v2 + C2))
res = []
for c in range(1, a.n + 1):
    lqim = Image.open(next(VAL.glob(f'case{c}_lq.*'))).convert('RGB')
    gtim = Image.open(next(VAL.glob(f'case{c}_gt.*'))).convert('RGB')
    W, H = lqim.size
    box = ((W - 512) // 2, (H - 512) // 2, (W - 512) // 2 + 512, (H - 512) // 2 + 512)
    lqc = np.asarray(lqim.crop(box), dtype=np.uint8); gtc = np.asarray(gtim.crop(box), dtype=np.uint8)
    t = torch.from_numpy(lqc.copy()).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
    o = restore(t)[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
    oc = np.clip((o + 1) * 127.5, 0, 255).astype(np.uint8)
    t2 = lambda z: torch.from_numpy(z.transpose(2, 0, 1)[None] / 127.5 - 1).float()
    with torch.no_grad():
        l_p = float(lp(t2(oc), t2(gtc)).item()); l_g = float(lp(t2(lqc), t2(gtc)).item())
    res.append((psnr(oc, gtc), ssim_np(oc, gtc), l_p, psnr(lqc, gtc), ssim_np(lqc, gtc), l_g))
    print(f'case{c}: PixRestore PSNR {res[-1][0]:.3f} (LQ {res[-1][3]:.3f}) | SSIM {res[-1][1]:.4f} (LQ {res[-1][4]:.4f}) '
          f'| LPIPS {l_p:.4f} (LQ {l_g:.4f})', flush=True)
m = np.array(res)
print(f'\n均值: PSNR {m[:,0].mean():.3f} | SSIM {m[:,1].mean():.4f} | LPIPS {m[:,2].mean():.4f}', flush=True)
print(f'[参照 M24 ] PSNR 24.559 | SSIM 0.9658 | LPIPS 0.4408  (val 512 中心裁切同口径)', flush=True)
print('PIXRESTORE_TEST_DONE')
