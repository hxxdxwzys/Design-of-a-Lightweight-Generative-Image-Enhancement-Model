# -*- coding: utf-8 -*-
"""TTA 自集成渲染 + 逐图择优 (140 阶梯可选步, 分支 A 后 GPU 空出时运行):
对给定选择映射中的每张图, 用其源 ckpt 渲染 8 向 TTA (flip/rot 平均),
与 plain (sweep 已渲染) 按 4K clipiqa 逐图比较, TTA 严格更优才替换; 输出最终目录。
无源 ckpt (旧 run 已删) 的图保持 plain。
用法: D:/Python/python.exe render_tta_choice.py --map M0.010 --out D:/models/cnn_out_mix_tta
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
from train_df2k import NAFNet  # noqa: E402
import pyiqa  # noqa: E402

TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PATCH, OVERLAP = 512, 128
SWEEP = Path(r'D:/models/sweep_test_out')

CKPT_MAP = {
    'old_s6000': Path(r'D:/models/metric_net/net_s6000.pt'),
}


def ckpt_for(name):
    if name in CKPT_MAP:
        return CKPT_MAP[name]
    for tag, d in [('f2k', Path(r'D:/models/metric_net_f2k')),
                   ('metric_net_v2b', Path(r'D:/models/metric_net_v2b'))]:
        if name.startswith(tag + '_'):
            stem = name[len(tag) + 1:]
            p = d / f'{stem}.pt'
            return p if p.exists() else None
    return None


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


def tiled(net, x, device, mask):
    H, W = x.shape[2], x.shape[3]
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
                out = net(p).clamp(0, 1)
                acc[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += out * mask
                wsum[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += mask
    return acc / wsum


def load_t(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='M0.010', help='choice_maps.json 的 M 键 (如 M0.010)')
    ap.add_argument('--out', default=r'D:/models/cnn_out_mix_tta')
    args = ap.parse_args()

    device = 'cuda'
    choice = json.load(open(r'D:/tmp_timing/choice_maps.json'))['maps'][args.map]['choice']
    sr = json.load(open(r'D:/models/sweep_test_out/sweep_results.json'))
    pi = {r['name']: r['per_image'] for r in sr['results'] if 'per_image' in r}
    keys = sorted(choice, key=lambda k: int(k.replace('case', '')))

    # 按源 ckpt 分组, 每 ckpt 只加载一次
    groups = {}
    for k in keys:
        src = choice[k]
        groups.setdefault(src, []).append(k)

    cq_m = pyiqa.create_metric('clipiqa', device=device)
    mask = make_mask().to(device)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    report = {}
    for src, ks in sorted(groups.items()):
        ckpt = ckpt_for(src)
        if ckpt is None:
            print(f'[skip] {src}: 无源 ckpt ({len(ks)} 图保持 plain)', flush=True)
            for k in ks:
                plain_dir = Path(r'D:/models/cnn_out_s6k') if src == 'old_s6000' else SWEEP / src
                shutil.copy2(plain_dir / f'{k}.jpg', out / f'{k}.jpg')
                report[k] = {'src': src, 'used': 'plain_no_ckpt'}
            continue
        net = NAFNet(width=32).to(device).eval()
        ck = torch.load(ckpt, map_location=device)
        net.load_state_dict(ck['net'] if 'net' in ck else ck)
        print(f'[load] {src} <- {ckpt.name} ({len(ks)} 图)', flush=True)
        for k in ks:
            plain_dir = Path(r'D:/models/cnn_out_s6k') if src == 'old_s6000' else SWEEP / src
            plain_p = plain_dir / f'{k}.jpg'
            plain_cq = pi.get(src, {}).get(k, {}).get('clipiqa')
            if plain_cq is None:
                x = load_t(plain_p).to(device).clamp(1e-4, 1 - 1e-4)
                plain_cq = float(cq_m(x).mean().item())
            x = load_t(TEST_DIR / f'{k}.jpg').to(device)
            acc = torch.zeros_like(x)
            for rot in range(4):
                for flip in (False, True):
                    xt = torch.rot90(x, rot, (2, 3))
                    if flip:
                        xt = torch.flip(xt, (3,))
                    o = tiled(net, xt, device, mask)
                    if flip:
                        o = torch.flip(o, (3,))
                    o = torch.rot90(o, -rot, (2, 3))
                    acc += o
            tta = acc / 8
            tta_cq = float(cq_m(tta.clamp(1e-4, 1 - 1e-4)).mean().item())
            use_tta = tta_cq > plain_cq
            arr = (tta[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8) \
                if use_tta else np.asarray(Image.open(plain_p).convert('RGB'))
            Image.fromarray(arr).save(out / f'{k}.jpg', quality=98)
            report[k] = {'src': src, 'used': 'tta' if use_tta else 'plain',
                         'plain_cq': round(float(plain_cq), 5), 'tta_cq': round(tta_cq, 5)}
            print(f'  {k}: plain={plain_cq:.4f} tta={tta_cq:.4f} -> {"TTA" if use_tta else "plain"}', flush=True)
        del net
        torch.cuda.empty_cache()

    n_tta = sum(1 for v in report.values() if v['used'] == 'tta')
    mean_plain = float(np.mean([v['plain_cq'] for v in report.values()]))
    mean_final = float(np.mean([(v['tta_cq'] if v['used'] == 'tta' else v['plain_cq']) for v in report.values()]))
    print(f'\n[summary] {len(report)} 图, TTA 采用 {n_tta}, plain 均值 {mean_plain:.4f} -> 最终 {mean_final:.4f}', flush=True)
    json.dump(report, open(r'D:/tmp_timing/tta_choice_report.json', 'w'), indent=1)
    print(f'[ok] {out}', flush=True)


if __name__ == '__main__':
    main()
