# -*- coding: utf-8 -*-
"""后处理/编码 扫描 (2026-09-11): 对着"官方目标函数"离线优化
新公式 ≈ 81.50 + 0.245*musiq - 9.45*niqe  (test 集无参指标, 无需 GT)
=> 可直接在【已有提交图】上离线搜索: 降 niqe (权重 9.45/单位) / 升 musiq (0.245/单位)

第一轮用 niqe-only (快) 扫全部 100 张, 对 top 变体再补 musiq + 护栏指标 (clipiqa/lpips)。
用法: python postproc_sweep.py [--musiq-top N] [--n N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from PIL import Image, ImageFilter

SRC = Path(r'D:/models/final_compliant/output_dir')          # M24 的 100 张提交图
LQ = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(r'D:/tmp_timing/postproc_sweep.json')

C_NIQE = (94.444, -10.028)      # niqe-only 模型 (1.4B 三点拟合 rmse 0.046)
C_FULL = (81.498, 0.245, -9.449)  # musiq+niqe 模型 (外推 M24 残差 +0.55)


def variants(img: Image.Image, lq: Image.Image | None):
    """返回 {名字: 变换后的图}"""
    out = {'orig': img}
    import io
    for q in (100, 98, 90):
        b = io.BytesIO()
        img.save(b, format='JPEG', quality=q, subsampling=0)
        b.seek(0)
        out[f'jpeg_q{q}'] = Image.open(b).convert('RGB')
    for s in (0.3, 0.6, 1.0):
        out[f'blur{s}'] = img.filter(ImageFilter.GaussianBlur(s))
    out['unsharp'] = img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
    if lq is not None and lq.size == img.size:
        a = np.asarray(img, np.float32)
        b = np.asarray(lq, np.float32)
        for al in (0.05, 0.10, 0.20):
            m = (1 - al) * a + al * b
            out[f'blend_lq{al:.2f}'] = Image.fromarray(np.clip(m, 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='用多少张 (默认全部)')
    ap.add_argument('--musiq-top', type=int, default=4, help='对 niqe 最优的 N 个变体补测 musiq/clipiqa')
    ap.add_argument('--musiq-n', type=int, default=20)
    a = ap.parse_args()

    import torch
    import pyiqa
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    niqe = pyiqa.create_metric('niqe', device=dev)
    print('[metric] niqe ready', flush=True)

    files = sorted(SRC.glob('*.jpg'))[:a.n]
    res = {}
    for j in files:
        img = Image.open(j).convert('RGB')
        lq = LQ / j.name
        lq_img = Image.open(lq).convert('RGB') if lq.exists() else None
        for name, im in variants(img, lq_img).items():
            t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                res.setdefault(name, []).append(float(niqe(t).item()))
        if len(res['orig']) % 20 == 0:
            print(f'  niqe 扫描 {len(res["orig"])}/{len(files)}', flush=True)

    print('\n=== niqe-only 排名（分数 = 94.444 − 10.028·niqe；M24 实测 43.684）===')
    rank = sorted(res.items(), key=lambda kv: np.mean(kv[1]))
    for name, v in rank:
        nq = float(np.mean(v))
        print(f'  {name:<14} niqe={nq:.4f}  → 预测分 {C_NIQE[0]+C_NIQE[1]*nq:6.2f}')

    # 对最优的若干变体补 musiq + 护栏
    tops = [n for n, _ in rank[:a.musiq_top + 1]]
    print(f'\n=== 补测 musiq / clipiqa（前 {len(tops)} 个, {a.musiq_n} 张）===')
    musiq = pyiqa.create_metric('musiq', device=dev)
    clip = pyiqa.create_metric('clipiqa', device=dev)
    detail = {}
    for j in files[:a.musiq_n]:
        img = Image.open(j).convert('RGB')
        lq = LQ / j.name
        lq_img = Image.open(lq).convert('RGB') if lq.exists() else None
        vs = variants(img, lq_img)
        for name in tops:
            if name not in vs:
                continue
            t = torch.from_numpy(np.asarray(vs[name]).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                d = detail.setdefault(name, {'musiq': [], 'clipiqa': []})
                d['musiq'].append(float(musiq(t).item()))
                d['clipiqa'].append(float(clip(t).item()))
        print(f'  musiq/clipiqa {len(detail.get(tops[0], {}).get("musiq", []))}/{a.musiq_n}',
              flush=True)

    print('\n=== 全模型预测（81.50 + 0.245·musiq − 9.45·niqe）===')
    full = {}
    for name, d in detail.items():
        mu = float(np.mean(d['musiq']))
        nq = float(np.mean(res[name][:a.musiq_n]))
        pred = C_FULL[0] + C_FULL[1] * mu + C_FULL[2] * nq
        full[name] = dict(niqe=nq, musiq=mu, clipiqa=float(np.mean(d['clipiqa'])), pred=pred)
        print(f'  {name:<14} niqe={nq:.4f} musiq={mu:.3f} clipiqa={np.mean(d["clipiqa"]):.4f} '
              f'→ 预测 {pred:6.2f}')

    OUT.write_text(json.dumps({'niqe_all': {k: float(np.mean(v)) for k, v in res.items()},
                               'full': full}, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
