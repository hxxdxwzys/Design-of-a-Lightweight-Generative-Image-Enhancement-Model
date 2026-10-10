# -*- coding: utf-8 -*-
"""用 PixRestore 出 4K 并测质量 (tile 512 + 羽化), 对比 M24 的 clipiqa 0.9023 / lapvar 508"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

REPO = Path(r'D:/models/newsrc/PixRestore_code')
sys.path.insert(0, str(REPO))
from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
from pixrestore.vision import extract_layers, load_dinov2

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='b'); ap.add_argument('--n', type=int, default=24)
ap.add_argument('--steps', type=int, default=1)
ap.add_argument('--out', default=r'D:/tmp_timing/pix_out')
a = ap.parse_args()
CKPT = Path(rf'D:/models/newsrc/PixRestore/pixrestore-{a.size}')
cfg = json.loads((CKPT / 'config.json').read_text(encoding='utf-8'))
cfg['dinov2_repository'] = r'D:/models/newsrc/dinov2_repo'
cfg['dinov2_checkpoint'] = str(Path(r'D:/models/newsrc/dinov2') /
    {'dinov2s':'dinov2_vits14_pretrain.pth','dinov2b':'dinov2_vitb14_pretrain.pth','dinov2l':'dinov2_vitl14_pretrain.pth'}[cfg['encoder_type']])
config = SimpleNamespace(**cfg)
dev = 'cuda'
model = LightningDiT_PixelDiffusion(
    input_size=config.resolution, patch_size=config.patch_size, in_channels=6, out_channels=3,
    hidden_size=config.hidden_size, depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
    z_dims=config.encoder_dim if config.use_venc else None, encdim_ratio=config.encoder_dim_ratio,
    use_qknorm=config.use_qknorm, use_swiglu=config.use_swiglu, use_rope=config.use_rope,
    use_rmsnorm=config.use_rmsnorm, adain_single=True, num_fused_layers=len(config.encoder_layers),
    pca_dim=config.bottleneck_dim, use_bottleneck_patch_embed=True,
    use_dino_layer_router=config.use_dino_layer_router, dino_layer_router_mode=config.dino_layer_router_mode,
    same_condition_layer_weights=config.same_condition_layer_weights, gate_temperature=config.gate_temperature,
    feature_norm=config.feature_norm, use_lq_degradation_token=bool(getattr(config,'use_lq_degradation_token',False)),
    register_local_gate_logit_scale=bool(getattr(config,'register_local_gate_logit_scale',False)))
model.load_state_dict(load_file(str(CKPT/'clean_weights'/'ema_model.safetensors')), strict=False)
model = model.to(dev).eval().requires_grad_(False)
flow = PixelDiffusion(flow_ratio=config.flow_ratio, time_dist=config.time_distribution, alpha=config.alpha,
                      z_start='noise', cfg_ratio=config.cfg_ratio, cfg_scale=config.cfg_scale,
                      image_size=config.resolution, channels=3, interp_type='lin', uncond_type='zero',
                      norm_p=config.adaptive_loss_power, accelerator=None, t_start=0, t_end=1, use_cos=False, args=config)
enc = load_dinov2(config.encoder_type, dev, repository=config.dinov2_repository, checkpoint=config.dinov2_checkpoint)
print(f'[model] pixrestore-{a.size}', flush=True)

TILE, OVER = 512, 32
mask = np.ones((TILE, TILE, 1), np.float32); r = OVER//2
ramp = np.linspace(0,1,r,dtype=np.float32)
mask[:r,:,0]*=ramp[:,None]; mask[-r:,:,0]*=ramp[::-1,None]; mask[:,:r,0]*=ramp[None,:]; mask[:,-r:,0]*=ramp[::-1][None,:]

@torch.no_grad()
def run(patch):
    t = torch.from_numpy(patch.copy()).permute(2,0,1)[None].to(dev).float().div_(127.5).sub_(1)
    fea = extract_layers(enc, t, config.encoder_layers, config.encoder_input_size)
    with torch.autocast('cuda', dtype=torch.float16):
        o = flow.sample_multistep_fm(model, t, venc_fea=fea, n_steps=a.steps, schedule='linear')
    return np.clip((o[0].permute(1,2,0).float().cpu().numpy()+1)*127.5, 0, 255)

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
from scipy import ndimage
names = [f'case{i}' for i in range(1, 101)][:a.n]
lv = []
t0 = time.time()
for k, n in enumerate(names, 1):
    if (OUT / f'{n}.jpg').exists():
        continue          # 断点续跑
    img = np.asarray(Image.open(TEST/f'{n}.jpg').convert('RGB'), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H,W,3), np.float32); ws = np.zeros((H,W,1), np.float32)
    for y in sorted(set(list(range(0, max(1,H-TILE+1), TILE-OVER)) + [max(0,H-TILE)])):
        for x in sorted(set(list(range(0, max(1,W-TILE+1), TILE-OVER)) + [max(0,W-TILE)])):
            acc[y:y+TILE, x:x+TILE] += run(img[y:y+TILE, x:x+TILE]) * mask
            ws[y:y+TILE, x:x+TILE] += mask
    out = np.clip(acc/np.maximum(ws,1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(OUT/f'{n}.jpg', quality=98)
    g = np.asarray(Image.fromarray(out).convert('L'), dtype=np.float32)
    lv.append(float(ndimage.laplace(g).var()))
    if k % 6 == 0:
        print(f'  {k}/{len(names)} {(time.time()-t0)/k:.1f}s/张', flush=True)
print(f'[lapvar] {np.mean(lv):.1f}  (M24 508 / TinyV2 110)', flush=True)
import pyiqa
m = pyiqa.create_metric('clipiqa', device=dev)
vs = []
for n in names:
    t = torch.from_numpy(np.asarray(Image.open(OUT/f'{n}.jpg').convert('RGB'))).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    with torch.no_grad(): vs.append(float(m(t).item()))
    del t
print(f'[clipiqa] {np.mean(vs):.4f}  (M24 0.9023 / v5 0.3897 / LQ 0.2752)', flush=True)
print('PIX_RENDER_DONE')
