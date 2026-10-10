# -*- coding: utf-8 -*-
"""FiDeSR 分块 4x 驱动 (2026-08-25): 4K 输入 -> 1024px 块 -> 各块原生 4x 推理 -> 4x 下采样 -> 拼接
动机: 1x 模式 clip 卡 0.65 (不在训练分布); 4x 是原生模式, 分块解决 16K 整图 OOM。
特性: 逐图幂等 (输出存在即跳过) + 暂停标记 (fidesr_pause.flag, 图像边界) + 逐图日志
用法: D:/Python/python.exe D:/tmp_timing/b_fidesr_tiled4x.py -i <dir|file> -o <out_dir> [--lf 0.2] [--hf 0.3] [--align wavelet|nofix]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
FD = r'D:/models/new_src/fidesr/code/FiDeSR-main'
SD = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
PKL = r'D:/models/new_src/fidesr/fidesr.pkl'
PAUSE = TT / 'fidesr_pause.flag'
TILE = 1024
UP = 4


def log(msg):
    print(msg, flush=True)
    with (TT / 'b_fidesr_tiled4x.log').open('a', encoding='utf-8') as f:
        f.write(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}\n')


def run_tile(tile_png, out_dir, lf, hf, align):
    """单块 4x 推理, 返回输出 png 路径"""
    cmd = [PY, 'test_fidesr.py', '--pretrained_model_path', SD, '--pretrained_path', PKL,
           '--input_image', str(tile_png), '--output_dir', str(out_dir),
           '--upscale', str(UP), '--align_method', align,
           '--lf_scale', str(lf), '--hf_scale', str(hf),
           '--mixed_precision', 'fp16', '--seed', '42']
    r = subprocess.run(cmd, cwd=FD, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return r.returncode


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--lf', type=float, default=0.2)
    ap.add_argument('--hf', type=float, default=0.3)
    ap.add_argument('--align', default='wavelet')
    args = ap.parse_args()

    from PIL import Image
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='fidesr_tile_'))

    inp = Path(args.input)
    imgs = sorted(inp.glob('*.jpg')) + sorted(inp.glob('*.png')) if inp.is_dir() else [inp]
    imgs = list(dict.fromkeys(imgs))
    log(f'[boot] {len(imgs)} images, lf={args.lf} hf={args.hf} align={args.align}')

    for img_path in imgs:
        stem = img_path.stem
        outp = out_dir / f'{stem}.png'
        if outp.exists():
            log(f'[skip] {stem} exists')
            continue
        if PAUSE.exists():
            log('[pause] flag found, clean stop')
            sys.exit(0)
        t0 = time.time()
        im = Image.open(img_path).convert('RGB')
        W, H = im.size
        # 对齐到 TILE 倍数 (边缘反射填充)
        Wp = ((W + TILE - 1) // TILE) * TILE
        Hp = ((H + TILE - 1) // TILE) * TILE
        if (Wp, Hp) != (W, H):
            pad_im = Image.new('RGB', (Wp, Hp))
            pad_im.paste(im, (0, 0))
            im = pad_im
        canvas = Image.new('RGB', (Wp, Hp))
        n_tiles = 0
        for y0 in range(0, Hp, TILE):
            for x0 in range(0, Wp, TILE):
                tile = im.crop((x0, y0, x0 + TILE, y0 + TILE))
                tp = tmp / f'{stem}_{y0}_{x0}.png'
                tile.save(tp)
                rc = run_tile(tp, str(tmp), args.lf, args.hf, args.align)
                if rc != 0:
                    log(f'[err] {stem} tile ({x0},{y0}) rc={rc}')
                    continue
                out4 = tmp / f'{stem}_{y0}_{x0}.png'
                if not out4.exists():
                    log(f'[err] {stem} tile output missing')
                    continue
                t4 = Image.open(out4)
                t1 = t4.resize((TILE, TILE), Image.LANCZOS)
                canvas.paste(t1, (x0, y0))
                n_tiles += 1
        canvas = canvas.crop((0, 0, W, H))
        canvas.save(outp)
        log(f'[done] {stem} ({n_tiles} tiles, {time.time()-t0:.0f}s)')
    log('[done] all')


if __name__ == '__main__':
    main()
