# -*- coding: utf-8 -*-
"""FiDeSR CPU 冒烟 (2026-08-25): 验证 0.39 兼容加载链, 不碰 GPU
步骤: SD2.1 unet 加载 -> 3 适配器 -> pkl LoRA/LRRB -> 小张量前向
用法: cd D:/models/new_src/fidesr/code/FiDeSR-main && D:/Python/python.exe D:/tmp_timing/b_fidesr_smoke.py
"""
import os
import sys

sys.path.insert(0, r'D:/models/new_src/fidesr/code/FiDeSR-main')
os.chdir(r'D:/models/new_src/fidesr/code/FiDeSR-main')

import torch
import torch.nn.functional as F
from types import SimpleNamespace

print('[1] imports...', flush=True)
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from fidesr import FiDeSR_eval, DenseBlock, LRRB

SD = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
PKL = r'D:/models/new_src/fidesr/fidesr.pkl'

print('[2] build UNet (sd2.1 config) + adapters + pkl...', flush=True)
args = SimpleNamespace(
    pretrained_model_path=SD, pretrained_path=PKL,
    lrrb_in_ch=8, lrrb_mid_ch=64, lrrb_growth=32, lrrb_res_sc=0.2,
    vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
    latent_tiled_size=96, latent_tiled_overlap=32,
    mixed_precision='fp16',
)
m = FiDeSR_eval(args)
print('[3] loaded. unet adapters:', [n for n, _ in m.unet.named_parameters() if 'lora' in n][:3], '...')
print('[4] unet lora params:', sum(p.numel() for n, p in m.unet.named_parameters() if 'lora' in n))
print('[5] lrrb params:', sum(p.numel() for p in m.lrrb.parameters()))

print('[6] tiny forward: unet(latent, t=1, prompt_embeds) + lrrb...', flush=True)
lat = torch.randn(1, 4, 16, 16).cuda().half()
pe = torch.randn(1, 77, 1024).cuda().half()
t1 = torch.tensor([1]).long().cuda()
with torch.no_grad():
    pred = m.unet(lat, t1, encoder_hidden_states=pe).sample
    delta = m.lrrb(torch.cat([lat, pred], dim=1))
    out = pred + delta
print('[7] forward OK:', tuple(out.shape), 'std=%.3f' % out.std().item())

print('[8] text encoder check...', flush=True)
from transformers import AutoTokenizer, CLIPTextModel
tok = AutoTokenizer.from_pretrained(SD, subfolder='tokenizer')
te = CLIPTextModel.from_pretrained(SD, subfolder='text_encoder')
ids = tok('', max_length=77, padding='max_length', truncation=True, return_tensors='pt').input_ids
with torch.no_grad():
    emb = te(ids)[0]
print('[9] text OK:', tuple(emb.shape))

print('[10] VAE encode/decode tiny...', flush=True)
x = torch.randn(1, 3, 64, 64).cuda().half()
with torch.no_grad():
    z = m.vae.encode(x).latent_dist.sample() * m.vae.config.scaling_factor
    r = m.vae.decode(z / m.vae.config.scaling_factor).sample
print('[11] VAE OK:', tuple(r.shape))

print('SMOKE_ALL_OK')
