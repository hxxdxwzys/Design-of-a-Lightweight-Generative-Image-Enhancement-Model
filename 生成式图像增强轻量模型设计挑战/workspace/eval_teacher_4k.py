"""v5 阶段1 (止损闸门): OSEDiff 教师对 3 张 4K 验证集 LQ 出图 -> eval.py 评测。

用途: 教师输出 clipiqa 显著 >0.35 才继续蒸馏, 否则止损。
教师 4K 整图前向在 V100 32G 会 OOM, 故分块 (512 tile + 64 重叠高斯融合),
每块过完整管线 (RAM tags + CLIP + UNet 单步)。

用法:
    python eval_teacher_4k.py --osediff_dir /data/osediff/OSEDiff \
        --sd_path /data/weights/sd21base --ram_path /data/weights/ram_swin_large_14m.pth \
        --input_dir /data/val/lq --output_dir /data/val/teacher_out
    然后: python eval.py --pred-dir /data/val/teacher_out --gt-dir /data/val/gt ...
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.osediff_teacher import OSEDiffTeacher
from infer import gaussian_weights  # 复用分块融合逻辑


@torch.no_grad()
def enhance_tiled(teacher, img_pil, tile=512, overlap=64):
    """教师分块推理: RAM 每块独立 (先跑通; 若 tag 质量差再改整图 tags)。"""
    img_pil = img_pil.convert('RGB')
    W, H = img_pil.size
    arr = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0
    pw = (tile - W % tile) % tile
    ph = (tile - H % tile) % tile
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    H2, W2 = arr.shape[:2]

    out = np.zeros((H2, W2, 3), dtype=np.float64)
    wsum = np.zeros((H2, W2, 1), dtype=np.float64)
    g = gaussian_weights(tile, overlap).numpy()[..., None]  # tile,tile,1
    step = tile - overlap
    for y in range(0, H2, step):
        for x in range(0, W2, step):
            y1 = min(y, H2 - tile)
            x1 = min(x, W2 - tile)
            patch = Image.fromarray(
                np.clip((arr[y1:y1 + tile, x1:x1 + tile] * 127.5 + 127.5),
                        0, 255).astype(np.uint8))
            enh = np.asarray(teacher.infer(patch), dtype=np.float64)
            out[y1:y1 + tile, x1:x1 + tile] += enh * g
            wsum[y1:y1 + tile, x1:x1 + tile] += g
    out = out / wsum.clip(min=1e-8)
    out = np.clip(out[:H, :W], 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osediff_dir', required=True)
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--ram_path', required=True)
    ap.add_argument('--dape_path', default=None)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = OSEDiffTeacher(args.osediff_dir, args.sd_path, args.ram_path,
                             args.dape_path, device, fp16=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*_lq.*')))
    print(f'val LQ images: {len(files)}', flush=True)
    for i, f in enumerate(files):
        img = Image.open(f)
        out = enhance_tiled(teacher, img, tile=args.tile, overlap=args.overlap)
        name = os.path.basename(f).replace('_lq', '_teacher')
        out.save(os.path.join(args.output_dir, name), quality=95)
        print(f'[{i+1}/{len(files)}] {name} -> {out.size}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
