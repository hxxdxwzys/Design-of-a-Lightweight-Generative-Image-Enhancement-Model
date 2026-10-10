# -*- coding: utf-8 -*-
"""测试集全 ckpt 直选 sweep (第 1 步诊断, 2026-08-14 deepseek冲刺4)

对旧 run (metric_net, 17 个) + f2k run (metric_net_f2k, 16 个) 全部 ckpt:
  1) 渲染 100 张测试集 LQ -> 4K q98 jpg (512 tile + 128 overlap 羽毛融合, fp32)
  2) 算 4K clipiqa + niqe 均值 (pyiqa, 与官方口径一致: (0,1) fp32 clamp; musiq 4K 太慢 42s/img 且官方权重≈0, 已砍)
  3) 增量写 sweep_results.json, 断点续跑 (每 ckpt 一个 .done 标记)

复用已有输出: old_s6000 -> D:/models/cnn_out_s6k ; f2k_s4000 -> D:/models/cnn_out_f2k
渲染顺序: s6k 邻域 (s5k-s10k) 优先, 便于尽早拿到关键比较信号。
脚本本身放工作区 (C:), 输入读 D:, 输出写 D:/models/sweep_test_out (需 danger-full-access)。

用法:
  D:/Python/python.exe sweep_test_all.py                        # 全量
  D:/Python/python.exe sweep_test_all.py --only f2k_s5000 --limit 3   # 冒烟
  D:/Python/python.exe sweep_test_all.py --no-render            # 只算指标
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
from train_df2k import NAFNet  # noqa: E402
import pyiqa  # noqa: E402

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PATCH = 512
OVERLAP = 128
BASE_OUT = Path(r'D:/models/sweep_test_out')
RESULTS_JSON = BASE_OUT / 'sweep_results.json'

REUSE = {
    'old_s6000': Path(r'D:/models/cnn_out_s6k'),
    'f2k_s4000': Path(r'D:/models/cnn_out_f2k'),
}


def build_jobs(extra_dirs=None, min_step=10000, max_step=10**9):
    jobs = []  # (name, ckpt_path, out_dir, reuse)
    old_dir = Path(r'D:/models/metric_net')
    f2k_dir = Path(r'D:/models/metric_net_f2k')
    if extra_dirs:
        for d in extra_dirs:
            d = Path(d)
            tag = d.name
            for p in sorted(d.glob('*.pt'), key=lambda x: x.name):
                m = re.match(r'(net_s|ema_s)(\d+)\.pt$', p.name)
                if m and min_step <= int(m.group(2)) <= max_step:
                    jobs.append((f'{tag}_{p.stem}', p, BASE_OUT / f'{tag}_{p.stem}', False))
    old_order = ['s5000', 's7000', 's8000', 's9000', 's10000']  # 2026-08-14 裁剪: s5k/s7k 已输 s6k, 尾部低 EV 跳过
    for s in old_order:
        p = old_dir / f'net_{s}.pt'
        if p.exists():
            jobs.append((f'old_{s}', p, BASE_OUT / f'old_{s}', False))
    for s in range(1000, 17000, 1000):
        if s == 4000:
            continue  # f2k_s4000 由 REUSE (cnn_out_f2k) 覆盖, 勿重复渲染
        p = f2k_dir / f'net_s{s}.pt'
        if p.exists():
            jobs.append((f'f2k_s{s}', p, BASE_OUT / f'f2k_s{s}', False))
    # 参考件 (复用已渲染输出, 只算指标)
    for name, d in REUSE.items():
        jobs.append((name, None, d, True))
    return jobs


def make_mask():
    m = torch.ones(PATCH, PATCH)
    ramp = OVERLAP // 2
    for i in range(ramp):
        w = (i + 1) / ramp
        m[i, :] = w
        m[PATCH - 1 - i, :] = w
        m[:, i] = torch.minimum(m[:, i], torch.full((PATCH,), w))
        m[:, PATCH - 1 - i] = torch.minimum(m[:, PATCH - 1 - i], torch.full((PATCH,), w))
    return m


def tiled_infer(net, img_pil, device, mask):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    acc = torch.zeros(1, 3, H, W, device=device)
    wsum = torch.zeros(1, 1, H, W, device=device)
    step = PATCH - OVERLAP
    ys = list(range(0, max(H - PATCH, 0) + 1, step))
    xs = list(range(0, max(W - PATCH, 0) + 1, step))
    if (H - PATCH) > ys[-1]:
        ys.append(H - PATCH)
    if (W - PATCH) > xs[-1]:
        xs.append(W - PATCH)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                p = x[:, :, y0:y0 + PATCH, x0:x0 + PATCH]
                out = net(p).clamp(0, 1)  # fp32 推理 (fp16 LayerNorm 平坦 patch NaN)
                acc[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += out * mask
                wsum[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += mask
    return acc / wsum


def metrics_for_dir(out_dir, clipiqa_m, niqe_m, device):
    vals = {'clipiqa': [], 'niqe': []}
    per_image = {}
    for p in sorted(out_dir.glob('*.jpg')):
        a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
        cq = float(clipiqa_m(x).mean().item())
        nq = float(niqe_m(x).mean().item())
        vals['clipiqa'].append(cq)
        vals['niqe'].append(nq)
        per_image[p.stem] = {'clipiqa': round(cq, 6), 'niqe': round(nq, 6)}
    n = len(vals['clipiqa'])
    return {
        'n': n,
        'test_clipiqa': round(float(np.mean(vals['clipiqa'])), 6),
        'test_niqe': round(float(np.mean(vals['niqe'])), 6),
        'per_image': per_image,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--ckpt-dirs', default=None, help='额外 ckpt 目录, 逗号分隔 (如 D:/models/metric_net_v2)')
    ap.add_argument('--min-step', type=int, default=10000, help='额外目录只扫 >=min-step 的 ckpt (增补早期 ckpt 用)')
    ap.add_argument('--max-step', type=int, default=10**9, help='额外目录只扫 <=max-step 的 ckpt')
    ap.add_argument('--width', type=int, default=32, help='NAFNet 宽度 (w48 目录用 48)')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--no-metrics', action='store_true')
    args = ap.parse_args()

    BASE_OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.backends.cudnn.benchmark = True

    jobs = build_jobs([d for d in args.ckpt_dirs.split(',') if d.strip()] if args.ckpt_dirs else None,
                      min_step=args.min_step, max_step=args.max_step)
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
        if not jobs:
            print(f'[err] --only {args.only} 未找到', flush=True)
            return

    done = {}
    results = []
    if RESULTS_JSON.exists():
        try:
            existing = json.load(open(RESULTS_JSON))
            for r in existing.get('results', []):
                done[r['name']] = r
            results = list(existing.get('results', []))  # 继承已有条目 (合并语义, 防丢数据)
        except Exception as e:
            print(f'[warn] 读取旧结果失败: {e}', flush=True)

    clipiqa_m = None
    niqe_m = None
    if not args.no_metrics:
        clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
        niqe_m = pyiqa.create_metric('niqe', device=device)

    net = NAFNet(width=args.width).to(device).eval()
    mask = make_mask().to(device)
    test_files = sorted(TEST_DIR.glob('*.jpg'))[:args.limit]
    print(f'[start] {len(jobs)} jobs, limit={args.limit}, test_files={len(test_files)}', flush=True)

    for name, ckpt_path, out_dir, reuse in jobs:
        if name in done and done[name].get('n', 0) >= args.limit:
            print(f'[skip] {name} 已有结果 (n={done[name].get("n")})', flush=True)
            continue
        t0 = time.time()

        if not reuse and not args.no_render:
            out_dir.mkdir(parents=True, exist_ok=True)
            marker = out_dir / '.done'
            existing_n = len(list(out_dir.glob('*.jpg')))
            if marker.exists() and existing_n >= args.limit:
                print(f'[render] {name} 已完成 ({existing_n} 张), 跳过渲染', flush=True)
            else:
                ck = torch.load(ckpt_path, map_location=device)
                net.load_state_dict(ck['net'])
                net.eval()
                step = ck.get('step', '?')
                print(f'[render] {name} step={step} 开始 (0/{len(test_files)})', flush=True)
                t_img0 = time.time()
                for i, p in enumerate(test_files):
                    img = Image.open(p).convert('RGB')
                    out = tiled_infer(net, img, device, mask)
                    arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(out_dir / f'{p.stem}.jpg', quality=98)
                    if (i + 1) % 10 == 0 or i + 1 == len(test_files):
                        el = time.time() - t_img0
                        print(f'  [{i + 1}/{len(test_files)}] {el / (i + 1):.2f}s/img, eta {el / (i + 1) * (len(test_files) - i - 1) / 60:.1f}min', flush=True)
                if args.limit >= 100:
                    marker.write_text(str(step))
                print(f'[render] {name} 完成 {(time.time() - t0) / 60:.1f}min', flush=True)

        if not args.no_metrics:
            m = metrics_for_dir(out_dir, clipiqa_m, niqe_m, device)
            rec = {'name': name, **m}
            print(f'[metrics] {name}: clipiqa={m["test_clipiqa"]:.4f} niqe={m["test_niqe"]:.2f} ({(time.time() - t0) / 60:.1f}min)', flush=True)
        else:
            rec = {'name': name}
            print(f'[done] {name} (no metrics)', flush=True)

        results = [r for r in results if r['name'] != name]
        results.append(rec)
        with open(RESULTS_JSON, 'w') as f:
            json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)

    # 兜底: 循环结束后再写一次完整 results (skip 分支的条目只在内存里, 必须在收尾落盘)
    with open(RESULTS_JSON, 'w') as f:
        json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)

    # 汇总
    with_metrics = [r for r in results if 'test_clipiqa' in r]
    with_metrics.sort(key=lambda r: -r['test_clipiqa'])
    print('\n===== 汇总 (按 test clipiqa 降序) =====', flush=True)
    for r in with_metrics:
        print(f"  {r['name']:14s} clipiqa={r['test_clipiqa']:.4f} niqe={r['test_niqe']:.2f}", flush=True)
    print('[all done]', flush=True)


if __name__ == '__main__':
    main()