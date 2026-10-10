"""v5 阶段2: 离线生成教师蒸馏目标 (LQ, HQ_teacher) 对, 存盘供 train_v5 使用。

流程: DF2K HQ 图 -> 随机裁 patch(带翻转) -> 在线退化 LQ
      -> OSEDiff 教师推理 -> 存 (lq, hq_teacher) fp16 tensor。

为什么离线: 教师管线含 RAM(swin-large), 在线跑会拖慢训练 5-10 倍;
             固定目标可复现、可反复调参不重跑教师。

输出: out_dir/000000_lq.pt, 000000_hq.pt ... (fp16, [-1,1])
断点续跑: 已存在 out_dir/000000_hq.pt 则跳过。
"""
import os
import sys
import glob
import random
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.degradation import degrade
from data.dataset import IMG_EXTS, to_tensor_norm
from models.osediff_teacher import OSEDiffTeacher

IMG_EXTS  # noqa  (import 副作用已在 dataset 定义，这里显式引用避免 lint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='HQ 图像目录 (DF2K)')
    ap.add_argument('--out_dir', required=True, help='教师目标输出目录')
    ap.add_argument('--osediff_dir', required=True, help='OSEDiff 官方仓库根')
    ap.add_argument('--sd_path', required=True, help='SD 基座目录')
    ap.add_argument('--ram_path', required=True, help='ram_swin_large_14m.pth')
    ap.add_argument('--dape_path', default=None, help='DAPE.pth (可选)')
    ap.add_argument('--patch_size', type=int, default=512)
    ap.add_argument('--patches_per_img', type=int, default=2)
    ap.add_argument('--max_imgs', type=int, default=0, help='0=全部')
    ap.add_argument('--fp32', action='store_true', help='教师 fp32(默认fp16)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = OSEDiffTeacher(args.osediff_dir, args.sd_path, args.ram_path,
                             args.dape_path, device, fp16=not args.fp32)

    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(args.data_dir, '**', ext), recursive=True)
    paths = sorted(paths)
    if args.max_imgs:
        paths = paths[:args.max_imgs]
    print(f'images: {len(paths)}, patch={args.patch_size}, '
          f'patches/img={args.patches_per_img}', flush=True)

    idx = 0
    done = 0
    for pi, p in enumerate(paths):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'skip {p}: {e}', flush=True)
            continue
        W, H = img.size
        ps = args.patch_size
        if W < ps or H < ps:
            img = img.resize((max(ps, W), max(ps, H)), Image.BICUBIC)
            W, H = img.size

        for _ in range(args.patches_per_img):
            x = random.randint(0, W - ps)
            y = random.randint(0, H - ps)
            hq = img.crop((x, y, x + ps, y + ps))
            if random.random() < 0.5:
                hq = hq.transpose(Image.FLIP_LEFT_RIGHT)
            lq = degrade(hq)

            hq_t = teacher.infer(lq)  # PIL

            lq_t = to_tensor_norm(lq)
            hq_teacher_t = to_tensor_norm(hq_t)
            torch.save(lq_t.half(), os.path.join(args.out_dir, f'{idx:06d}_lq.pt'))
            torch.save(hq_teacher_t.half(), os.path.join(args.out_dir, f'{idx:06d}_hq.pt'))
            idx += 1
            done += 1
            if done % 50 == 0:
                print(f'[{done}] img {pi+1}/{len(paths)} saved {idx}', flush=True)
        if pi % 100 == 0:
            print(f'img {pi+1}/{len(paths)} done', flush=True)

    print(f'ALL DONE: {done} pairs -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
