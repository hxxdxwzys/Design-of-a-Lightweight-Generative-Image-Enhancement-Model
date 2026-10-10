# -*- coding: utf-8 -*-
"""通用新源入池 (2026-08-17): 任意开源/自训生成器输出 -> 4K jpg -> 逐图 clipiqa/niqe -> 并入 sweep_results.json
用法:
  D:/Python/python.exe pool_add_source.py --src D:/models/codiff_out --name codiff
说明:
  - --src 内应有 case1..case100 输出 (任意扩展名 png/jpg/webp); 若尺寸 < 对应 LQ 4K 尺寸则 bicubic 放大对齐
  - 可选 --val-src: 3 张 val LQ 输出 (case1_lq/case2_lq/case3_lq 任意扩展名), 用于流形门控 (clipiqa + lpips vs GT)
  - 合并语义: 仅替换同名条目, 其余全部保留 (先备份 sweep_results.json)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402
from sweep_test_all import metrics_for_dir, BASE_OUT, RESULTS_JSON  # noqa: E402

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')


def lq_size(case_key):
    p = TEST_DIR / f'{case_key}.jpg'
    with Image.open(p) as im:
        return im.size  # (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='新源输出目录 (含 case1..100)')
    ap.add_argument('--name', required=True, help='池条目名 (如 codiff / instructir)')
    ap.add_argument('--val-src', default=None, help='val 3 张输出目录 (流形门控)')
    ap.add_argument('--skip-val-gate', action='store_true')
    args = ap.parse_args()

    src = Path(args.src)
    assert src.is_dir(), f'--src 不存在: {src}'

    # 1) 归一化到池目录
    dst = BASE_OUT / args.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for p in sorted(src.iterdir()):
        stem = p.stem
        key = stem.replace('_lq', '')
        if not key.startswith('case'):
            continue
        with Image.open(p) as im:
            im = im.convert('RGB')
            tw, th = lq_size(key)
            if im.size != (tw, th):
                print(f'[resize] {key}: {im.size} -> ({tw},{th})', flush=True)
                im = im.resize((tw, th), Image.BICUBIC)
            im.save(dst / f'{key}.jpg', quality=98)
            n += 1
    print(f'[convert] {args.name}: {n} jpgs -> {dst}', flush=True)
    if n < 100:
        print(f'[abort] 不足 100 张 ({n}), 不并入池', flush=True)
        return

    # 2) 备份 + 合并 (安全语义)
    bak = RESULTS_JSON.with_name(f'sweep_results.backup_{time.strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy2(RESULTS_JSON, bak)
    print(f'[backup] {bak}', flush=True)

    device = 'cuda'
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    niqe_m = pyiqa.create_metric('niqe', device=device)
    m = metrics_for_dir(dst, clipiqa_m, niqe_m, device)
    rec = {'name': args.name, **m}

    d = json.load(open(RESULTS_JSON))
    results = [r for r in d.get('results', []) if r['name'] != args.name]
    results.append(rec)
    json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(RESULTS_JSON, 'w'), indent=1)
    print(f'[merged] {args.name}: clipiqa={m["test_clipiqa"]:.4f} niqe={m["test_niqe"]:.2f}, entries={len(results)}', flush=True)

    # 3) 对照报告
    by_name = {r['name']: r for r in results}
    vosr = by_name.get('vosr_05b_os')
    if vosr:
        wins = sum(1 for k, v in rec['per_image'].items()
                   if v['clipiqa'] > vosr['per_image'].get(k, {}).get('clipiqa', 0))
        gains = [v['clipiqa'] - vosr['per_image'][k]['clipiqa'] for k, v in rec['per_image'].items() if k in vosr['per_image']]
        print(f'[vs vosr_05b_os] 胜场 {wins}/100, 平均增量 {np.mean(gains):+.4f}', flush=True)

    # 4) val 流形门控 (可选)
    if args.val_src and not args.skip_val_gate:
        vsrc = Path(args.val_src)
        gt_files = {p.stem.replace('_gt', ''): p for p in VAL_DIR.glob('*_gt.*')}
        lpips_m = pyiqa.create_metric('lpips', device=device)
        val_clip, val_lp = [], []
        for name, gtp in sorted(gt_files.items()):
            lqp = VAL_DIR / f'{name}_lq.jpg'
            for cand in sorted(vsrc.iterdir()):
                if cand.stem.startswith(name) and '_lq' in cand.stem:
                    with Image.open(cand) as im:
                        out = torch.from_numpy(np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
                                               ).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
                    with Image.open(gtp) as im:
                        gt = torch.from_numpy(np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
                                              ).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
                    val_clip.append(float(clipiqa_m(out).mean().item()))
                    val_lp.append(float(lpips_m(out, gt).mean().item()))
                    break
        if val_clip:
            print(f'[val gate] clipiqa={np.mean(val_clip):.3f} (VOSR 0.629) | '
                  f'lpips={np.mean(val_lp):.3f} (VOSR 0.342) | '
                  f'PASS={np.mean(val_clip) >= 0.55 and np.mean(val_lp) <= 0.372}', flush=True)

    print('[done]', flush=True)


if __name__ == '__main__':
    main()
