#!/bin/bash
# P100 服务器导出 M24 合规 JIT 模型 (2026-09-02): torch 2.4.1 jit.trace (无 dynamo 冲突)
# 前置: symlink preset -> weights (dinov2 离线补丁的相对路径), cwd=~/p100
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
export TORCHDYNAMO_SUPPRESS_ERRORS=1
P="$HOME/p100"
mkdir -p "$P/preset/ckpts"
[ -e "$P/preset/ckpts/torch_cache" ] || ln -s "$P/weights/torch_cache" "$P/preset/ckpts/torch_cache"

cd "$P"
python - <<'PYEOF'
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/p100/VOSR'))
sys.path.insert(0, os.path.expanduser('~/p100/tools'))

import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL
from models.lightningdit import LightningDiT
from models.light_decoder import LightDecoder
from inference_vosr_onestep import load_config_with_cli, load_dinov2

P = os.path.expanduser('~/p100')
CKPT = os.path.join(P, 'weights/VOSR_0.5B_os')
SD2_AE = os.path.join(P, 'weights/sd21base')
LWDEC = os.path.join(P, 'weights/sd21_lwdecoder.pth')
LORA = os.path.join(P, 'tools/v7/lora_s5000.pt')
LORA_SCALE = 0.8
OUT = os.path.join(P, 'your_model.pt')
CACHE = os.path.join(P, 'weights/torch_cache')

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', '-c', default=CKPT)
ap.add_argument('--config', default='')
args = load_config_with_cli(CKPT, ap)
if getattr(args, 'auxiliary_time_cond', None) is None:
    args.auxiliary_time_cond = True
args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
torch.hub.set_dir(CACHE)

# diT + LoRA sc0.8
model = LightningDiT(
    input_size=args.resolution // 8, patch_size=args.patch_size,
    in_channels=2 * 4, out_channels=4, hidden_size=args.dim, depth=args.depth,
    num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim,
    encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
    use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
    use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
    num_fused_layers=len(args.layer_dinov2b_list))
sd = load_file(os.path.join(CKPT, 'checkpoints/ema_model.safetensors'))
model.load_state_dict(sd, strict=False)
model.forward = model.forward_flexible
lora = torch.load(LORA, map_location='cpu', weights_only=False)
cfg = json.loads(lora['lora_config'])
model = get_peft_model(model, LoraConfig(**cfg))
lora_sd = {k: v for k, v in lora['state_dict'].items() if 'lora' in k}
if LORA_SCALE != 1.0:
    lora_sd = {k: (v * LORA_SCALE if 'lora_B' in k else v) for k, v in lora_sd.items()}
model.load_state_dict(lora_sd, strict=False)
model = model.half().cuda().eval()

vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').half().cuda().eval()
lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
dec = LightDecoder(**{k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
dec.load_state_dict(lw['model_state_dict'])
dec = dec.half().cuda().eval()
venc = load_dinov2(args, 'cuda').half().eval()
print('[ok] 组装完成', flush=True)

MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1).cuda()
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1).cuda()
SCALE = vae.config.scaling_factor

class FullChain(torch.nn.Module):
    def forward(self, x, z):
        with torch.autocast('cuda', dtype=torch.float16):
            raw = torch.nn.functional.interpolate((0.5 * x + 0.5) * 255.0,
                                                  size=args.dinov2_size, mode='bicubic').clamp(0., 255.)
            xn = (raw / 255.0 - MEAN) / STD
            feats, x_norm = venc.forward_with_features(xn)
            zz = [v for k, v in feats.items() if k.startswith('layer_')]
            zz[-1] = x_norm
            zz = [zz[i] for i in args.layer_dinov2b_list]
            lat = vae.encode(x).latent_dist.sample() * SCALE
            u = model(torch.cat([lat, z], 1), torch.ones(x.shape[0], device=x.device),
                      torch.zeros(x.shape[0], device=x.device), zz)
            z1 = z - u
            sr = dec(z1 / SCALE).clamp(-1, 1)
        return sr

chain = FullChain().cuda().eval()
for p in chain.parameters():
    p.requires_grad_(False)
x = (torch.rand(1, 3, 512, 512).cuda() * 2 - 1).half()
z = torch.randn(1, 4, 64, 64).cuda().half()
with torch.no_grad():
    ref = chain(x, z)
print(f'[eager] std={ref.float().std().item():.3f} mean={ref.float().mean().item():.3f}', flush=True)
with torch.no_grad():
    traced = torch.jit.trace(chain, (x, z), check_trace=False)
with torch.no_grad():
    out = traced(x, z)
d = ref.float() - out.float()
print(f'[trace] std={out.float().std().item():.3f} MAE={d.abs().mean().item():.5f}', flush=True)
assert out.float().std().item() > 0.05
assert d.abs().mean().item() < 0.15
torch.jit.save(traced, OUT)
print(f'[save] {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)', flush=True)
for _ in range(3):
    traced(x, z)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    traced(x, z)
torch.cuda.synchronize()
print(f'[speed] 512 全链 {(time.time()-t0)/10*1000:.1f} ms (P100)', flush=True)
print('EXPORT_DONE', flush=True)
PYEOF
