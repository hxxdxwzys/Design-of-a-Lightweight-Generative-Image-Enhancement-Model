# -*- coding: utf-8 -*-
"""P3: 测试集 100 张 4K 渲染 (参数化版)
用法: python render_test100.py --lora <ckpt> --scale <float> --tag <输出目录名>
输出: D:/models/final_compliant_<tag>/{render_png, output_dir}/
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
    ROOT = Path(r'D:/models') / f'final_compliant_{args.tag}'
    PNG_DIR = ROOT / 'render_png'
    OUT_DIR = ROOT / 'output_dir'
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG = ROOT / 'render_test100.log'

    def log(m):
        line = time.strftime('[%m-%d %H:%M:%S] ') + m
        print(line, flush=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    jpgs = sorted(TEST.glob('case*.jpg'))
    log(f'[boot] 渲染 {len(jpgs)} 张 (tag={args.tag} scale={args.scale})')
    t0 = time.time()
    done = 0
    for j in jpgs:
        stem = j.stem  # caseN
        out_png = PNG_DIR / f'{stem}.png'
        out_jpg = OUT_DIR / f'{stem}.jpg'
        if out_jpg.exists():
            done += 1
            continue
        if not out_png.exists():
            t1 = time.time()
            ok = False
            for attempt in range(2):
                try:
                    r = subprocess.run(
                        [sys.executable, r'D:/tmp_timing/vosr_lora_infer_14b.py',
                         '-c', BASE, '--lora', args.lora, '--lora-scale', str(args.scale), '--fp16',
                         '-i', str(j), '-o', str(PNG_DIR), '--align_method', 'wavelet'],
                        capture_output=True, text=True, timeout=900)
                    if r.returncode == 0:
                        ok = True
                        break
                    log(f'[FAIL] {stem} attempt{attempt + 1}: {r.stderr[-200:]}')
                except subprocess.TimeoutExpired:
                    log(f'[TIMEOUT] {stem} attempt{attempt + 1} (900s)')
            if not ok:
                log(f'[SKIP] {stem} 渲染失败, 跳过')
                continue
            log(f'[渲染] {stem} ({time.time()-t1:.0f}s)')
        # png -> jpg q95
        from PIL import Image
        img = Image.open(out_png).convert('RGB')
        img.save(out_jpg, quality=95)
        done += 1
        log(f'[jpg] {stem} -> {out_jpg.name}')
    log(f'[done] {done}/{len(jpgs)} 张, 耗时 {(time.time()-t0)/60:.1f} min')
    print('RENDER_TEST100_DONE' if done == len(jpgs) else f'PARTIAL {done}/{len(jpgs)}')


if __name__ == '__main__':
    main()
