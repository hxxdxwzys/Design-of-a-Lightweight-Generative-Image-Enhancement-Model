"""v4 JIT 导出: 加载 V4 + lora_state + merge LoRA + 移除 text_encoder, trace (fp16) at 512。
用法: python export_jit_v4.py <sd_path> <ckpt> <out.pt>"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model_v4 import LightLatentDiffV4


def main():
    sd, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    device = 'cuda'
    ck = torch.load(ckpt, map_location='cpu')
    model = LightLatentDiffV4(sd, lora_rank=ck.get('lora_rank', 16))
    model.unet.load_state_dict(ck['lora_state'], strict=False)
    model.unet = model.unet.merge_and_unload()  # 合并 LoRA
    # 推理只用 vae+unet+prompt_embeds(buffer), 移除 text_encoder/scheduler 减体积
    model._modules.pop('text_encoder', None)
    model._modules.pop('scheduler', None)
    model = model.to(device).eval().half()  # fp16
    example = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(out)
    with torch.no_grad():
        o = traced(example)
    print(f'saved JIT -> {out} | {tuple(o.shape)} {o.dtype} NaN={torch.isnan(o).any().item()} '
          f'range=[{o.min().item():.3f},{o.max().item():.3f}]', flush=True)


if __name__ == '__main__':
    main()
