#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v5 阶段2 (官方 venv 版): 离线生成 OSEDiff 教师蒸馏目标。

用官方环境 (D:/venv_osediff311, diffusers 0.25) 加载 OSEDiff_test,
对 DF2K HQ 图随机裁 patch -> 在线退化 LQ -> OSEDiff 教师推理 -> 存 (lq, hq_t, hq_gt)。

输出: out_dir/000000_lq.pt + _hq_t.pt (教师目标) + _hq_gt.pt (GT锚定), fp16 [-1,1]
断点续跑: 已存在 _hq_t.pt 则跳过。

用法 (官方 venv):
  D:/venv_osediff311/Scripts/python.exe gen_teacher_targets_official.py \
    --data_dir D:/datasets/div2k --out_dir D:/models/teacher_targets \
    --patch_size 512 --patches_per_img 2 --max_imgs 2000
"""
import os
import sys
import glob
import random
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, 'D:/models/OSEDiff-main/OSEDiff-main')
from osediff import OSEDiff_test
from data.degradation import degrade
from data.dataset import IMG_EXTS, to_tensor_norm

PROMPT = 'a photo of a high-quality, sharp, clean scene,'

def make_osediff_args(sd_path, osediff_dir):
    class _Args:
        pass
    a = _Args()
    a.pretrained_model_name_or_path = sd_path
    a.osediff_path = os.path.join(osediff_dir, 'preset/models/osediff.pkl')
    a.mixed_precision = 'fp16'
    a.merge_and_unload_lora = False
    a.latent_tiled_size = 96
    a.latent_tiled_overlap = 32
    a.vae_encoder_tiled_size = 512
    a.vae_decoder_tiled_size = 224
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='D:/datasets/div2k', help='HQ 图像目录')
    ap.add_argument('--out_dir', default='D:/models/teacher_targets')
    ap.add_argument('--sd_path', default='D:/models/sd21base')
    ap.add_argument('--osediff_dir', default='D:/models/OSEDiff-main/OSEDiff-main')
    ap.add_argument('--patch_size', type=int, default=512)
    ap.add_argument('--patches_per_img', type=int, default=2)
    ap.add_argument('--max_imgs', type=int, default=0, help='0=全部')
    ap.add_argument('--no_ram', action='store_true', help='跳过 RAM, 用固定 prompt')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    model = OSEDiff_test(make_osediff_args(args.sd_path, args.osediff_dir)).to(device).eval()

    # RAM 标签 (可选)
    ram = None
    if not args.no_ram:
        ram_path = 'D:/models/ram_swin_large_14m.pth'
        dape_path = os.path.join(args.osediff_dir, 'preset/models/DAPE.pth')
        if os.path.exists(ram_path):
            try:
                from ram.models.ram_lora import ram as ram_cls
                from ram import inference_ram as inference
                from ram import get_transform
                ram = ram_cls(pretrained=ram_path, pretrained_condition=dape_path,
                              image_size=384, vit='swin_l').to(device).eval().half()
                ram_tf = get_transform(image_size=384)
                print('[ram] loaded', flush=True)
            except Exception as e:
                print(f'[ram] skip: {e}', flush=True)
                ram = None

    def get_prompt(img_pil):
        if ram is not None:
            try:
                t = ram_tf(img_pil.convert('RGB')).unsqueeze(0).to(device).half()
                caps = inference(t, ram)
                if caps and caps[0]:
                    return f'{caps[0]}, {PROMPT},'
            except Exception:
                pass
        return PROMPT

    # 收集图像
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(args.data_dir, '**', ext), recursive=True)
    paths = sorted(paths)
    if args.max_imgs:
        paths = paths[:args.max_imgs]
    print(f'images: {len(paths)}, patch={args.patch_size}, patches/img={args.patches_per_img}', flush=True)

    idx = 0
    done = 0
    ps = args.patch_size
    for pi, p in enumerate(paths):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'skip {p}: {e}', flush=True)
            continue
        W, H = img.size
        if W < ps or H < ps:
            img = img.resize((max(ps, W), max(ps, H)), Image.BICUBIC)
            W, H = img.size
        for _ in range(args.patches_per_img):
            out_lq = os.path.join(args.out_dir, f'{idx:06d}_lq.pt')
            out_t = os.path.join(args.out_dir, f'{idx:06d}_hq_t.pt')
            out_gt = os.path.join(args.out_dir, f'{idx:06d}_hq_gt.pt')
            if os.path.exists(out_t):
                idx += 1
                continue
            x = random.randint(0, W - ps)
            y = random.randint(0, H - ps)
            hq = img.crop((x, y, x + ps, y + ps))
            if random.random() < 0.5:
                hq = hq.transpose(Image.FLIP_LEFT_RIGHT)
            lq = degrade(hq)
            prompt = get_prompt(lq)
            with torch.no_grad():
                hq_t = model(to_tensor_norm(lq).unsqueeze(0).to(device), prompt).clamp(-1, 1)
            lq_t = to_tensor_norm(lq)
            hq_gt_t = to_tensor_norm(hq)
            torch.save(lq_t.half(), out_lq)
            torch.save(hq_t.squeeze(0).cpu().half(), out_t)
            torch.save(hq_gt_t.half(), out_gt)
            idx += 1
            done += 1
            if done % 50 == 0:
                print(f'[{done}] img {pi+1}/{len(paths)} saved {idx}', flush=True)
        if pi % 100 == 0:
            print(f'img {pi+1}/{len(paths)} done', flush=True)
    print(f'ALL DONE: {done} pairs -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
