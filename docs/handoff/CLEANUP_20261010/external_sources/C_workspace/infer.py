"""分块推理：对任意尺寸(4K)图像按 tile 切块带重叠高斯融合，输出增强图。
用于生成 100 张 4K 测试结果。推理用 fp16 autocast (契合赛题)。
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff
from models.tiny_vae import TinyVAE


def gaussian_weights(tile, overlap):
    w = np.ones((tile, tile), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 2)[1:-1]
        for i in range(overlap):
            v = ramp[i]
            w[i, :] = np.minimum(w[i, :], v)
            w[-1 - i, :] = np.minimum(w[-1 - i, :], v)
            w[:, i] = np.minimum(w[:, i], v)
            w[:, -1 - i] = np.minimum(w[:, -1 - i], v)
    return torch.from_numpy(w)


@torch.no_grad()
def enhance_image(model, img_pil, tile=512, overlap=64, device='cuda', fp16=True):
    img_pil = img_pil.convert('RGB')
    W, H = img_pil.size
    arr = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0  # HWC, [-1,1]
    pw = (tile - W % tile) % tile
    ph = (tile - H % tile) % tile
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    H2, W2 = arr.shape[:2]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H2,W2

    out = torch.zeros(1, 3, H2, W2, device=device)
    wsum = torch.zeros(1, 1, H2, W2, device=device)
    g = gaussian_weights(tile, overlap).to(device)
    step = tile - overlap
    model.eval()
    for y in range(0, H2, step):
        for x in range(0, W2, step):
            y1 = min(y, H2 - tile)
            x1 = min(x, W2 - tile)
            patch = t[:, :, y1:y1 + tile, x1:x1 + tile]
            if fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    enh = model.forward(patch)
            else:
                enh = model.forward(patch)
            enh = enh.float()
            out[:, :, y1:y1 + tile, x1:x1 + tile] += enh * g
            wsum[:, :, y1:y1 + tile, x1:x1 + tile] += g
    out = out / wsum.clamp(min=1e-8)
    out = out[:, :, :H, :W]
    out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)


def load_model(sd_path, ckpt_path, device='cuda', tiny_vae=False):
    ck = torch.load(ckpt_path, map_location='cpu')
    unet_ch = ck.get('unet_ch', (64, 128, 256, 512))
    model = LightLatentDiff(sd_path, unet_ch=unet_ch)
    # 优先 EMA 权重 (v3+ 训练产物); 旧 ckpt 无 ema_unet 则回退 unet
    sd = ck.get('ema_unet', ck['unet'])
    model.unet.load_state_dict(sd)
    if tiny_vae:
        # 用 TAESD 轻量 VAE 替换 SD2.1 VAE: 时延 285ms->27ms, 加速比 6.8x
        model.vae = TinyVAE(device=device, fp16=True)
    model = model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--tiny_vae', action='store_true',
                    help='用 TAESD 轻量 VAE 替换 SD2.1 VAE (时延 285->27ms, 加速比 6.8x)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.sd_path, args.ckpt, device, tiny_vae=args.tiny_vae)
    print(f'loaded ckpt step={torch.load(args.ckpt,map_location="cpu").get("step","?")}', flush=True)

    import glob
    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
        except Exception as e:
            print(f'skip {f}: {e}', flush=True)
            continue
        out = enhance_image(model, img, tile=args.tile, overlap=args.overlap,
                            device=device, fp16=not args.fp32)
        out.save(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.jpg'), quality=95)
        print(f'[{i+1}/{len(files)}] {os.path.basename(f)} -> {out.size}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
