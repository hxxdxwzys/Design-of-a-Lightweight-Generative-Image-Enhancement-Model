# -*- coding: utf-8 -*-
"""CODiff 诊断: 官方 forward vs 分块路径, 同一 512x512 裁剪块 (cwd=仓库根)
"""
import sys
import os
import torch
import numpy as np
from PIL import Image

REPO = r'D:/models/finals/codiff/CODiff-main/CODiff-main'
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffusion.codiff import CODiff_test
from cave.cave import CaVE


class A:
    pass


args = A()
args.pretrained_model = r'D:/models/sd21base'
args.cave_path = r'D:/models/finals/codiff/cave.pth'
args.codiff_path = r'D:/models/finals/codiff/codiff.pkl'
args.vae_encoder_tiled_size = 1024
args.vae_decoder_tiled_size = 512
args.latent_tiled_size = 96
args.latent_tiled_overlap = 32
args.mixed_precision = 'fp16'
args.merge_and_unload_lora = False

model = CODiff_test(args)
cave = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
cave.load_state_dict(torch.load(args.cave_path), strict=True)
cave.eval()
cave = cave.to('cuda')

img = Image.open(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集/case1_lq.jpg').convert('RGB')
# 裁 512x512
crop = img.crop((1000, 800, 1512, 1312))
arr = np.asarray(crop).astype(np.float32) / 255.0
lq_raw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to('cuda')
lq = lq_raw * 2 - 1
print('lq_raw mean', float(lq_raw.mean()), 'std', float(lq_raw.std()))

print('alphas_cumprod[499] =', float(model.noise_scheduler.alphas_cumprod[499]))
print('alphas_cumprod[498] =', float(model.noise_scheduler.alphas_cumprod[498]))

with torch.no_grad():
    # 1) 官方 forward (VAEHook 路径)
    emb = cave.get_visual_embedding(lq_raw)
    out_official = model(lq, emb)
    print('official out mean', float(out_official.mean()), 'std', float(out_official.std()))

    # 2) 手动一步 (无 hook, 同一 CaVE emb)
    if hasattr(model.vae.encoder, 'original_forward'):
        model.vae.encoder.forward = model.vae.encoder.original_forward
    if hasattr(model.vae.decoder, 'original_forward'):
        model.vae.decoder.forward = model.vae.decoder.original_forward
    lat = model.vae.encode(lq_raw).latent_dist.sample() * model.vae.config.scaling_factor
    print('lat std', float(lat.std()))
    pred = model.unet(lat.to(torch.float16), torch.tensor([499], device='cuda').long(),
                      encoder_hidden_states=model.proj(emb).to(torch.float16)).sample
    print('pred std', float(pred.std()), 'pred mean', float(pred.mean()))
    a_t = model.noise_scheduler.alphas_cumprod[499].to('cuda')
    x0 = (lat - torch.sqrt(1 - a_t) * pred.float()) / torch.sqrt(a_t)
    print('x0 std', float(x0.std()), 'x0 mean', float(x0.mean()))
    dec = model.vae.decode((x0 / model.vae.config.scaling_factor).to(torch.float32)).sample.clamp(-1, 1)
    print('manual out mean', float(dec.mean()), 'std', float(dec.std()))
