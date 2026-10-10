# -*- coding: utf-8 -*-
"""1.4B 合规 pkg 合并 (参数化版): VOSR_1.4B_os + <lora> x <scale> -> your_model.pt
用法: python merge_14b.py --lora <ckpt> --scale <float> --tag <目录名>
输出: D:/models/final_compliant_<tag>/model_dir/your_model.pt (14b_compliant_v1 pkg)
"""
import argparse
import json
import os
import sys

os.chdir(r'C:/models/finals')  # _hub_load_offline_first 的相对路径依赖
sys.path.insert(0, r'D:/models/finals/VOSR')
sys.path.insert(0, r'D:/models/finals/VOSR/models')

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from models.lightningdit import LightningDiT  # noqa: E402
from models.qwenimage_vae2d import AutoencoderKLQwenImage2D  # noqa: E402
from inference_vosr_onestep import load_config_with_cli, load_dinov2  # noqa: E402

CKPT = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
QWEN_AE = r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d'

ap0 = argparse.ArgumentParser()
ap0.add_argument('--lora', required=True)
ap0.add_argument('--scale', type=float, default=1.0)
ap0.add_argument('--tag', required=True)
a0 = ap0.parse_args()
LORA = a0.lora
LORA_SCALE = a0.scale
OUT = rf'D:/models/final_compliant_{a0.tag}/model_dir/your_model.pt'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import argparse  # noqa: E402
sys.argv = [sys.argv[0]]  # 清 argv, 防 load_config_with_cli 内部 parse_args 读入新参数
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', '-c', default=CKPT)
ap.add_argument('--config', default='')
args = load_config_with_cli(CKPT, ap)
if getattr(args, 'auxiliary_time_cond', None) is None:
    args.auxiliary_time_cond = True
args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
args.enc_type = 'dinov2l'  # 1.4B 用 dinov2l (enc_dim 1024)
torch.hub.set_dir(r'C:/models/finals/preset/ckpts/torch_cache')

base_ch = 16
diT_cfg = dict(
    input_size=args.resolution // 8, patch_size=args.patch_size,
    in_channels=2 * base_ch, out_channels=base_ch, hidden_size=args.dim,
    depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
    z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio,
    auxiliary_time_cond=args.auxiliary_time_cond,
    use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu,
    use_rope=args.use_rope, use_rmsnorm=args.use_rmsnorm,
    wo_shift=args.wo_shift,
    num_fused_layers=len(args.layer_dinov2b_list))
model = LightningDiT(**diT_cfg)
sd = load_file(os.path.join(CKPT, 'checkpoints', 'ema_model.safetensors'))
model.load_state_dict(sd, strict=False)
model.forward = model.forward_flexible
lora = torch.load(LORA, map_location='cpu', weights_only=False)
cfg = json.loads(lora['lora_config'])
model = get_peft_model(model, LoraConfig(**cfg))
lora_sd = {k: v for k, v in lora['state_dict'].items() if 'lora' in k}
if LORA_SCALE != 1.0:
    lora_sd = {k: (v * LORA_SCALE if 'lora_B' in k else v) for k, v in lora_sd.items()}
model.load_state_dict(lora_sd, strict=False)
model = model.merge_and_unload() if hasattr(model, 'merge_and_unload') else model
model = model.eval()
print(f'[diT] merged {len(lora_sd)} lora keys @sc{LORA_SCALE}', flush=True)

vae = AutoencoderKLQwenImage2D.from_pretrained(QWEN_AE).eval()
venc = load_dinov2(args, 'cpu').eval()
venc.head = torch.nn.Identity()

vae_cfg = dict(
    base_dim=vae.base_dim if hasattr(vae, 'base_dim') else vae.config.base_dim,
    z_dim=vae.z_dim,
    dim_mult=list(vae.config.dim_mult) if hasattr(vae, 'config') else None,
    num_res_blocks=vae.config.num_res_blocks,
    attn_scales=list(vae.config.attn_scales),
    temperal_downsample=list(vae.config.temperal_downsample),
    dropout=float(vae.config.dropout),
    latents_mean=list(vae.config.latents_mean),
    latents_std=list(vae.config.latents_std),
    non_linearity=vae.config.non_linearity,
)
print(f'[vae] qwen 16ch base_dim={vae_cfg["base_dim"]}', flush=True)
print(f'[venc] dinov2l keys={len(venc.state_dict())}', flush=True)


def to_fp16(sd_dict):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd_dict.items()}


pkg = {
    'format': '14b_compliant_v1',
    'diT_cfg': diT_cfg,
    'diT': to_fp16(model.state_dict()),
    'vae': to_fp16(vae.state_dict()),
    'vae_cfg': vae_cfg,
    'venc': to_fp16(venc.state_dict()),
    'dinov2_size': args.dinov2_size,
    'layer_dinov2b_list': args.layer_dinov2b_list,
    'lora_scale': LORA_SCALE,
    'z_dim': base_ch,
}
torch.save(pkg, OUT)
print(f'[save] {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)', flush=True)
print('diT keys:', len(pkg['diT']), '| vae keys:', len(pkg['vae']), '| venc keys:', len(pkg['venc']), flush=True)
print('MERGE_14B_DONE', flush=True)
