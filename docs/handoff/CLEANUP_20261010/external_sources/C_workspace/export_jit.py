"""导出 JIT 模型 (torch.jit.trace, fp16) 用于提交测速。全模型含 VAE+UNet+显式x0。
用法: python export_jit.py <sd_path> <ckpt> <out.pt>
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff


def main():
    sd, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda"
    model = LightLatentDiff(sd).to(device).eval()
    ck = torch.load(ckpt, map_location="cpu")
    model.unet.load_state_dict(ck["unet"])
    model = model.half()  # fp16 (赛题要求)
    example = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(out)
    with torch.no_grad():
        o = traced(example)
    print(f"saved JIT -> {out} | out {tuple(o.shape)} {o.dtype} "
          f"NaN={torch.isnan(o).any().item()} range=[{o.min().item():.3f},{o.max().item():.3f}]", flush=True)


if __name__ == "__main__":
    main()
