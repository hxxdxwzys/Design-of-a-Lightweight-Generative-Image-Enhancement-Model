"""v4 分块推理: 加载 V4(SD UNet+LoRA) + lora_state + merge LoRA, 512 分块 4K 出图。
复用 infer.py 的 enhance_image/gaussian_weights。"""
import os
import sys
import argparse
import glob
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model_v4 import LightLatentDiffV4
from infer import enhance_image  # 复用分块+高斯融合


def load_v4_model(sd_path, ckpt_path, device='cuda'):
    ck = torch.load(ckpt_path, map_location='cpu')
    model = LightLatentDiffV4(sd_path, lora_rank=ck.get('lora_rank', 16))
    model.unet.load_state_dict(ck['lora_state'], strict=False)
    # 不合并 LoRA(diffusers UNet 无 merge_and_unload); 保留 LoRA 推理, 开销极小
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    model = load_v4_model(args.sd_path, args.ckpt, device)
    print(f'loaded v4 ckpt step={torch.load(args.ckpt, map_location="cpu").get("step","?")}', flush=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
        except Exception as e:
            print(f'skip {f}: {e}'); continue
        out = enhance_image(model, img, tile=args.tile, overlap=args.overlap, device=device, fp16=True)
        out.save(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.jpg'), quality=95)
        print(f'[{i+1}/{len(files)}] {os.path.basename(f)} -> {out.size}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
