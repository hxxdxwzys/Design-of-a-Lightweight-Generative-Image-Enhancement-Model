#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地评测闭环：生成式图像增强轻量模型设计挑战（赛道二）

计算有参考(PSNR/SSIM/LPIPS/DISTS)与无参考(CLIPIQA/MUSIQ/MANIQA/NIQE)指标，
按 case 号匹配 pred/gt，输出逐项与均值，保存 JSON。

用法示例（LQ 基线）:
  python eval.py \
      --pred-dir ../赛题二/验证集 --pred-pattern '*_lq.*' \
      --gt-dir   ../赛题二/验证集 --gt-pattern   '*_gt.*' \
      --out results/lq_baseline.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pyiqa

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REF_METRICS = ["psnr", "ssim", "lpips", "dists"]          # 需 GT
NOREF_METRICS = ["clipiqa", "musiq", "maniqa", "niqe"]     # 仅 pred
HIGHER_BETTER = {"psnr", "ssim", "clipiqa", "musiq", "maniqa"}


def case_key(name: str):
    m = re.search(r"case(\d+)", name)
    return int(m.group(1)) if m else None


def load_image(path, max_side=None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        w, h = im.size
        s = max_side / max(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W


def collect(dirpath, pattern="*"):
    d = {}
    for p in sorted(Path(dirpath).glob(pattern)):
        if p.suffix.lower() in IMG_EXTS:
            k = case_key(p.name)
            if k is not None:
                d[k] = p
    return d


def safe_score(fn):
    try:
        return float(fn().mean().item())
    except Exception as e:
        print(f"  [metric error] {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--pred-pattern", default="*")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--gt-pattern", default="*")
    ap.add_argument("--out", default=None, help="JSON 输出路径")
    ap.add_argument("--max-side", type=int, default=None,
                    help="长边缩放(加速); None=原分辨率(4K)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--metrics", default="all", help="逗号分隔指标名或 all")
    args = ap.parse_args()

    preds = collect(args.pred_dir, args.pred_pattern)
    gts = collect(args.gt_dir, args.gt_pattern) if args.gt_dir else {}
    keys = sorted(preds)
    if not keys:
        print("未找到 pred 图像，请检查 --pred-dir / --pred-pattern", file=sys.stderr)
        sys.exit(1)

    if args.metrics == "all":
        ref = REF_METRICS if gts else []
        noref = NOREF_METRICS
    else:
        sel = set(args.metrics.split(","))
        ref = [m for m in REF_METRICS if m in sel]
        noref = [m for m in NOREF_METRICS if m in sel]

    print(f"pred: {len(keys)} 张 | gt: {len(gts)} 张 | device: {args.device} | "
          f"max_side: {args.max_side or '原分辨率'}")
    print(f"有参: {ref or '无'} | 无参: {noref}")

    metrics = {}
    for m in ref + noref:
        try:
            metrics[m] = pyiqa.create_metric(m, device=args.device)
            print(f"  [metric] {m} 就绪")
        except Exception as e:
            print(f"  [warn] 无法创建指标 {m}: {e}", file=sys.stderr)

    results = {}
    for k in keys:
        pred = load_image(preds[k], args.max_side).to(args.device)
        rec = {}
        if gts and k in gts:
            gt = load_image(gts[k], args.max_side).to(args.device)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            for m in ref:
                if m in metrics:
                    rec[m] = safe_score(lambda m=m, p=pred, g=gt: metrics[m](p, g))
        for m in noref:
            if m in metrics:
                rec[m] = safe_score(lambda m=m, p=pred: metrics[m](p))
        results[k] = rec
        parts = [f"{m}={rec[m]:.4f}" if rec.get(m) is not None else f"{m}=NA"
                 for m in ref + noref]
        print(f"case{k:>3}: " + "  ".join(parts))

    agg = {}
    for m in ref + noref:
        vals = [results[k][m] for k in keys if results[k].get(m) is not None]
        if vals:
            agg[m] = sum(vals) / len(vals)

    print("\n=== 均值 ===")
    for m in ref + noref:
        if m in agg:
            tag = "↑越高越好" if m in HIGHER_BETTER else "↓越低越好"
            print(f"  {m:10s} {agg[m]:.4f}  {tag}")

    out = {"per_case": {str(k): v for k, v in results.items()},
           "mean": agg, "n": len(keys),
           "config": {"max_side": args.max_side, "device": args.device}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
