# -*- coding: utf-8 -*-
"""M24 合规模型导出 (2026-09-02): VOSR 0.5B + v7_s5000 LoRA(sc0.8) 全链 JIT trace
输出: 单个 JIT 模块 f(x[1,3,512,512], z[1,4,64,64]) -> sr[-1,1], fp16
用于规则收紧后的合规 model_dir (同一模型主体, 真计算测速)
"""
import os
import sys
import time

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'

sys.path.insert(0, r'D:/models/finals/VOSR')
sys.path.insert(0, r'D:/tmp_timing')

import argparse  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from models.lightningdit import LightningDiT  # noqa: E402
from models.light_decoder import LightDecoder  # noqa: E402
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, get_venc_features,
)

CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
SD2_AE = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
LWDEC = r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
LORA_SCALE = 0.8
OUT = r'D:/models/final_compliant/model_dir/your_model.pt'


class FullChain(torch.nn.Module):
    """VAE encode + dinov2 feats + diT one-step + LightDecoder decode (512 整图, 非分块)
    全链手动 fp16 (不依赖 autocast, trace 更稳)"""

    def __init__(self, diT, vae, dec, venc, cfg):
        super().__init__()
        self.diT = diT
        self.vae = vae
        self.dec = dec
        self.venc = venc
        self.cfg = cfg  # args namespace
        self.scale = vae.config.scaling_factor
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1)

    def venc_feats(self, x):
        """dinov2 特征 (全 fp16, 与出图语义一致: layer_dinov2b_list=[8])"""
        raw = torch.nn.functional.interpolate(
            (0.5 * x + 0.5) * 255.0, size=self.cfg.dinov2_size, mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)
        features, x_norm = self.venc.forward_with_features(xn)
        z = [v for k, v in features.items() if k.startswith('layer_')]
        z[-1] = x_norm
        z = [z[i] for i in self.cfg.layer_dinov2b_list]
        return z

    def forward(self, x, z):
        # autocast 与出图路径一致 (time embedding 等 fp32 中间计算由 autocast 管理)
        with torch.autocast('cuda', dtype=torch.float16):
            z_fea = self.venc_feats(x)
            lat = self.vae.encode(x).latent_dist.sample() * self.scale
            u = self.diT(torch.cat([lat, z], 1), torch.ones(x.shape[0], device=x.device),
                         torch.zeros(x.shape[0], device=x.device), z_fea)
            z1 = z - u
            sr = self.dec(z1 / self.scale).clamp(-1, 1)
        return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', '-c', default=CKPT)
    ap.add_argument('--config', default='')
    args = load_config_with_cli(CKPT, ap)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
    torch.hub.set_dir(r'C:/models/finals/preset/ckpts/torch_cache')

    # diT + LoRA(sc0.8)
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
    lora = torch.load(LORA, map_location='cpu', weights_only=False)
    cfg = json.loads(lora['lora_config'])
    model = get_peft_model(model, LoraConfig(**cfg))
    lora_sd = {k: v for k, v in lora['state_dict'].items() if 'lora' in k}
    if LORA_SCALE != 1.0:
        lora_sd = {k: (v * LORA_SCALE if 'lora_B' in k else v) for k, v in lora_sd.items()}
    model.load_state_dict(lora_sd, strict=False)
    model = model.half().cuda().eval()

    # VAE + LightDecoder + dinov2 (fp16)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').half().cuda().eval()
    lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
    dec = LightDecoder(**{k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
    dec.load_state_dict(lw['model_state_dict'])
    dec = dec.half().cuda().eval()
    venc = load_dinov2(args, 'cuda').half().eval()

    chain = FullChain(model, vae, dec, venc, args).cuda().eval()
    # torch 2.7: dynamo 自动优化与 jit.trace 冲突 -> 显式禁用全链 dynamo
    try:
        import torch._dynamo as _dyn
        for m in chain.modules():
            if hasattr(m, 'forward') and not isinstance(m.forward, torch.jit.ScriptFunction):
                try:
                    m.forward = _dyn.disable(m.forward)
                except Exception:
                    pass
    except Exception as e:
        print(f'[warn] dynamo disable failed: {e}', flush=True)
    print('[ok] 组装完成', flush=True)

    # eager 验证 + trace (vae 采样随机 -> 统计级验证)
    x = (torch.rand(1, 3, 512, 512).cuda() * 2 - 1).half()
    z = torch.randn(1, 4, 64, 64).cuda().half()
    with torch.no_grad():
        ref = chain(x, z)
    print(f'[eager] out std={ref.float().std().item():.3f} mean={ref.float().mean().item():.3f} '
          f'min={ref.float().min().item():.2f} max={ref.float().max().item():.2f}', flush=True)
    traced = torch.jit.trace(chain, (x, z), check_trace=False)
    with torch.no_grad():
        out = traced(x, z)
    d = ref.float() - out.float()
    print(f'[trace] out std={out.float().std().item():.3f} | eager-trace MAE={d.abs().mean().item():.5f} '
          f'(vae 采样噪声 ~0.01-0.05 属正常)', flush=True)
    assert out.float().std().item() > 0.05, 'trace 输出异常'
    assert d.abs().mean().item() < 0.15, 'trace 与 eager 差异过大'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.jit.save(traced, OUT)
    print(f'[save] {OUT} ({os.path.getsize(OUT) / 1e9:.2f} GB)', flush=True)

    # 简单计时
    for _ in range(3):
        traced(x, z)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        traced(x, z)
    torch.cuda.synchronize()
    print(f'[speed] 512 全链 {((time.time() - t0) / 10) * 1000:.1f} ms', flush=True)
    print('EXPORT_DONE', flush=True)


if __name__ == '__main__':
    main()
