"""v4 冒烟测试: 实例化 SD UNet+LoRA, 前向/反向, 查 NaN/显存/参数量。"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model_v4 import LightLatentDiffV4


def main():
    sd = sys.argv[1] if len(sys.argv) > 1 else '/data/weights/sd21base'
    device = torch.device('cuda')
    model = LightLatentDiffV4(sd).to(device)
    print(f'V4 trainable params: {model.param_count()/1e6:.2f}M', flush=True)

    x = torch.randn(1, 3, 512, 512, device=device).clamp(-1, 1)
    with torch.no_grad():
        out = model.forward(x)
    print(f'forward {x.shape}->{out.shape} NaN={torch.isnan(out).any().item()} '
          f'range=[{out.min().item():.3f},{out.max().item():.3f}]', flush=True)

    out2, x0, lq_lat = model(x, return_x0=True)
    loss = out2.float().mean() + x0.float().mean()
    loss.backward()
    g = next(p for p in model.trainable_params() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f'backward grad_abs_max={float(g.grad.abs().max()):.4f}', flush=True)
    print(f'GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB', flush=True)
    print('SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
