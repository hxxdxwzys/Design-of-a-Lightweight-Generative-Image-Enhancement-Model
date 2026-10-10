"""模型冒烟测试：实例化 + 前向 + 反向 + NaN/参数量检查。需 SD2.1 VAE 已下载。"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff


def main():
    sd = sys.argv[1] if len(sys.argv) > 1 else '/data/weights/sd21base'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightLatentDiff(sd).to(device)
    print(f'UNet params: {model.param_count()/1e6:.2f}M', flush=True)

    x = torch.randn(1, 3, 256, 256, device=device).clamp(-1, 1)
    with torch.no_grad():
        out = model.forward(x)
    print(f'forward: {x.shape} -> {out.shape} range [{out.min().item():.3f},{out.max().item():.3f}] '
          f'NaN={torch.isnan(out).any().item()}', flush=True)

    # 反向测试 (梯度经 VAE.decode -> scheduler.step -> UNet)
    x2 = torch.randn(1, 3, 256, 256, device=device).clamp(-1, 1)
    out2, x0, lq_lat = model(x2, return_x0=True)
    loss = out2.float().mean() + x0.float().mean()
    loss.backward()
    g = next(model.unet.parameters()).grad
    print(f'backward ok: grad={g is not None} grad_abs_max={float(g.abs().max()) if g is not None else 0:.4f}', flush=True)
    print('SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
