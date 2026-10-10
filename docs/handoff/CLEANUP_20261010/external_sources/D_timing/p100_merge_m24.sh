#!/bin/bash
# P100 服务器: 合并 M24 权重为 your_model.pt (fp16 state dicts + config, 供自包含 runner)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
export TORCHDYNAMO_SUPPRESS_ERRORS=1
P="$HOME/p100"
mkdir -p "$P/preset/ckpts"
[ -e "$P/preset/ckpts/torch_cache" ] || ln -s "$P/weights/torch_cache" "$P/preset/ckpts/torch_cache"

cd "$P"
python - <<'PYEOF'
import os, sys, json
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

# diT 结构参数 (runner 重建用)
diT_cfg = dict(
    input_size=args.resolution // 8, patch_size=args.patch_size,
    in_channels=2 * 4, out_channels=4, hidden_size=args.dim, depth=args.depth,
    num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim,
    encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
    use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
    use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
    num_fused_layers=len(args.layer_dinov2b_list))
model = LightningDiT(**diT_cfg)
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
# 合并 LoRA 进 base (runner 无需 peft)
model = model.merge_and_unload() if hasattr(model, 'merge_and_unload') else model
model = model.eval()

vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').eval()
lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
dec_cfg = {k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')}
dec = LightDecoder(**dec_cfg)
dec.load_state_dict(lw['model_state_dict'])
dec = dec.eval()
venc = load_dinov2(args, 'cpu').eval()
venc.head = torch.nn.Identity()

def to_fp16(sd_dict):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd_dict.items()}

pkg = {
    'format': 'm24_compliant_v1',
    'diT_cfg': diT_cfg,
    'diT': to_fp16(model.state_dict()),
    'vae': to_fp16(vae.state_dict()),
    'dec': to_fp16(dec.state_dict()),
    'venc': to_fp16(venc.state_dict()),
    'vae_cfg': vae.config.to_dict() if hasattr(vae.config, 'to_dict') else dict(vae.config),
    'dec_cfg': dec_cfg,
    'dinov2_size': args.dinov2_size,
    'layer_dinov2b_list': args.layer_dinov2b_list,
    'scale_factor': vae.config.scaling_factor,
    'lora_scale': LORA_SCALE,
}
torch.save(pkg, OUT)
print(f'[save] {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)', flush=True)
print('diT keys:', len(pkg['diT']), '| vae keys:', len(pkg['vae']),
      '| venc keys:', len(pkg['venc']), '| dec keys:', len(pkg['dec']), flush=True)
print('MERGE_DONE', flush=True)
PYEOF
