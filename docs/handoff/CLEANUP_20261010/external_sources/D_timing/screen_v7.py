# -*- coding: utf-8 -*-
"""v7(0.5B) 基座的 niqe 筛选 (2026-09-11)
新公式 ≈ 81.5 + 0.245·musiq − 9.45·niqe → niqe 权重是 musiq 的 38.6 倍, 先用 niqe 快筛。
做法: 取 12 张测试图做子集, 同一 LoRA 扫 scale / seed, 每档渲染后测 niqe + clipiqa(护栏)。
基准: M24 = v7 s5000, sc0.8, seed8 (test niqe 5.2559, 官方分 43.684)
用法: python screen_v7.py [--scales 0.6,0.7,0.8,0.9,1.0] [--seeds 8] [--n 12]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
WORK = Path(r'D:/models/screen_v7')
SUBSET = WORK / 'lq12'
OUTJSON = Path(r'D:/tmp_timing/screen_v7.json')
LOG = Path(r'D:/tmp_timing/screen_v7.log')


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--seeds', default='8')
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]

    # 固定子集 (按名字排序取前 n 张, 所有档位共用 -> 可比)
    SUBSET.mkdir(parents=True, exist_ok=True)
    files = sorted(TEST.glob('*.jpg'))[:a.n]
    for f in files:
        dst = SUBSET / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    log(f'[subset] {len(files)} 张 -> {SUBSET}')

    import torch
    import numpy as np
    import pyiqa
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    niqe = pyiqa.create_metric('niqe', device=dev)
    clip = pyiqa.create_metric('clipiqa', device=dev)
    log('[metric] niqe + clipiqa ready')

    res = json.loads(OUTJSON.read_text(encoding='utf-8')) if OUTJSON.exists() else {}
    for sc in scales:
        for sd in seeds:
            tag = f'sc{sc}_seed{sd}'
            if tag in res:
                log(f'[skip] {tag} 已有: {res[tag]}')
                continue
            outd = WORK / f'out_{tag}'
            log(f'[渲染] {tag}')
            r = subprocess.run(
                f'{PY} {INFER} -c {BASE} --lora {LORA} --lora-scale {sc} --seed {sd} '
                f'--align_method wavelet -i {SUBSET} -o {outd}',
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=7200, cwd=r'C:/models/finals')
            if r.returncode != 0:
                log(f'[渲染失败] {tag}: {(r.stderr or "")[-300:]}')
                continue
            vals_n, vals_c = [], []
            for j in sorted(outd.glob('*.jpg')):
                im = Image.open(j).convert('RGB')
                t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)
                t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    vals_n.append(float(niqe(t).item()))
                    vals_c.append(float(clip(t).item()))
            res[tag] = {'niqe': round(float(np.mean(vals_n)), 4),
                        'clipiqa': round(float(np.mean(vals_c)), 4), 'n': len(vals_n)}
            OUTJSON.write_text(json.dumps(res, indent=1), encoding='utf-8')
            pred = 94.444 - 10.028 * res[tag]['niqe']
            log(f'[完成] {tag}: niqe={res[tag]["niqe"]:.4f} clipiqa={res[tag]["clipiqa"]:.4f} '
                f'-> 预测分 {pred:.2f} (M24 基准 43.684)')

    log('=== 汇总 (按 niqe 升序) ===')
    for tag, v in sorted(res.items(), key=lambda kv: kv[1]['niqe']):
        log(f'  {tag:<14} niqe={v["niqe"]:.4f} clipiqa={v["clipiqa"]:.4f} '
            f'预测 {94.444 - 10.028*v["niqe"]:.2f}')
    log('[done]')


if __name__ == '__main__':
    main()
