# -*- coding: utf-8 -*-
"""VOSR 0.5B os 真模型 512x512 测速 (2026-09-02, 规则三合规测速件基础数据)
测量: 完整推理链 (VAE encode + dinov2 + DiT 1步 + LightDecoder decode) 的时延
"""
import os
import sys
import time

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ.setdefault('VOSR_CACHE', r'C:/models/finals/preset/ckpts/torch_cache')

VOSR_DIR = r'D:/models/finals/VOSR'
sys.path.insert(0, VOSR_DIR)
sys.path.insert(0, r'D:/tmp_timing')

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, get_venc_features, wavelet_color_fix,
)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
SD2_AE = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
LWDEC = r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth'
LORA = sys.argv[1] if len(sys.argv) > 1 else None  # 可选 LoRA
sys.argv = sys.argv[:1]  # 避免 load_config_with_cli 吃掉多余参数


def build():
    import argparse
    from models.lightningdit import LightningDiT
    from models.light_decoder import LightDecoder
    from diffusers import AutoencoderKL
    from peft import LoraConfig, get_peft_model
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', '-c', default=CKPT)
    ap.add_argument('--config', default='')
    args = load_config_with_cli(CKPT, ap)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
    torch.hub.set_dir(r'C:/models/finals/preset/ckpts/torch_cache')
    model = LightningDiT(
        input_size=args.resolution // 8, patch_size=args.patch_size,
        in_channels=2 * 4, out_channels=4, hidden_size=args.dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
        use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list))
    sd = load_file(os.path.join(CKPT, 'checkpoints', 'ema_model.safetensors'))
    model.load_state_dict(sd, strict=False)
    model.forward = model.forward_flexible
    if LORA:
        lora = torch.load(LORA, map_location='cpu', weights_only=False)
        cfg = __import__('json').loads(lora['lora_config'])
        model = get_peft_model(model, LoraConfig(**cfg))
        lora_sd = {k: v for k, v in lora['state_dict'].items() if 'lora' in k}
        model.load_state_dict(lora_sd, strict=False)
        print(f'[lora] {LORA} step={lora.get("step")}', flush=True)
    model = model.half().cuda().eval()
    vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').cuda().eval()
    lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
    dec = LightDecoder(**{k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
    dec.load_state_dict(lw['model_state_dict'])
    dec = dec.half().cuda().eval()
    venc = load_dinov2(args, 'cuda').half()
    return model, vae, dec, venc, args


def main():
    torch.manual_seed(0)
    model, vae, dec, venc, args = build()
    x = (torch.rand(1, 3, 512, 512).cuda() * 2 - 1)
    z = torch.randn(1, 4, 64, 64).cuda().half()

    def one_pass():
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                z_fea = get_venc_features(venc, x, args)
                z_fea = [z.detach() for z in z_fea]
                lat = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
                u = model(torch.cat([lat, z], 1), torch.ones(1, device='cuda'),
                          torch.zeros(1, device='cuda'), z_fea)
                z1 = z - u
                out = dec(z1 / vae.config.scaling_factor).clamp(-1, 1)
        return out

    # warmup
    for _ in range(3):
        one_pass()
    torch.cuda.synchronize()
    # 正式计时 (CPU-in: 计时含张量创建; GPU-in: 纯 kernel)
    ts = []
    for _ in range(10):
        t0 = time.time()
        one_pass()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f'512x512 全链时延: 中位 {med * 1000:.1f} ms | 最快 {ts[0] * 1000:.1f} ms | 10次: {[round(t * 1000, 1) for t in ts]}', flush=True)
    print(f'帧率: {1 / med:.1f} fps', flush=True)


if __name__ == '__main__':
    main()
