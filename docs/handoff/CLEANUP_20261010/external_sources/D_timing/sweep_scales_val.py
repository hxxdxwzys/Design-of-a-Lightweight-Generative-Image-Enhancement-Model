# -*- coding: utf-8 -*-
"""val 集 scale 扫描 (2026-09-11): 3 张验证图 x 多个 LoRA scale -> 全 8 指标
目的: 画出"保真(psnr/ssim/lpips/dists) vs 感知(clipiqa/musiq/niqe/maniqa)"权衡曲线,
      检验"高保真弱增强"假说, 并挑出探针候选。
成本: 每档 3 张 4K 渲染(~3min) + 8 指标评测(~5min)。
用法: python sweep_scales_val.py [--scales 0.0,0.2,0.4,0.6,0.8,1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROXY = 'http://127.0.0.1:7890'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = PROXY
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
EVAL = r'D:/tmp_timing/../生成式图像增强轻量模型设计挑战/workspace/eval.py'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
WORK = Path(r'D:/models/sweep_scale_val')
OUTJ = Path(r'D:/tmp_timing/sweep_scale_val.json')
LOG = Path(r'D:/tmp_timing/sweep_scale_val.log')


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='0.0,0.2,0.4,0.6,0.8,1.0')
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(',')]
    res = json.loads(OUTJ.read_text(encoding='utf-8')) if OUTJ.exists() else {}

    # val 的 LQ 输入目录
    lqdir = WORK / 'lq'
    lqdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(VAL.glob('*_lq.*')):
        dst = lqdir / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
    log(f'[val] LQ {len(list(lqdir.glob("*")))} 张')

    for sc in scales:
        tag = f'sc{sc}'
        if tag in res:
            log(f'[skip] {tag} 已有'); continue
        outd = WORK / f'pred_{tag}'
        log(f'[渲染] {tag}')
        r = subprocess.run(
            f'{PY} {INFER} -c {BASE} --lora {LORA} --lora-scale {sc} --seed 8 '
            f'--align_method wavelet -i {lqdir} -o {outd}',
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=7200, cwd=r'C:/models/finals')
        if r.returncode != 0:
            log(f'[渲染失败] {tag}: {(r.stderr or "")[-300:]}'); continue
        js = WORK / f'eval_{tag}.json'
        log(f'[评测] {tag}')
        subprocess.run(
            f'{PY} {EVAL} --pred-dir {outd} --pred-pattern "*_lq.*" '
            f'--gt-dir {VAL} --gt-pattern "*_gt.*" --out {js}',
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
        try:
            d = json.loads(js.read_text(encoding='utf-8'))
            res[tag] = d.get('mean', d)
            OUTJ.write_text(json.dumps(res, indent=1), encoding='utf-8')
            m = res[tag]
            log(f'[完成] {tag}: psnr={m.get("psnr")} ssim={m.get("ssim")} '
                f'lpips={m.get("lpips")} dists={m.get("dists")} clip={m.get("clipiqa")} '
                f'musiq={m.get("musiq")} niqe={m.get("niqe")} maniqa={m.get("maniqa")}')
        except Exception as e:
            log(f'[评测读取失败] {tag}: {e}')

    log('=== 汇总 (val, 3 张) ===')
    keys = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq', 'niqe', 'maniqa']
    log('  ' + 'scale'.ljust(8) + ''.join(k[:8].rjust(10) for k in keys))
    for tag, m in sorted(res.items(), key=lambda kv: kv[0]):
        row = tag.ljust(8)
        for k in keys:
            v = m.get(k)
            row += ('     -    ' if v is None else f'{v:10.4f}')
        log('  ' + row)
    log('[done]')


if __name__ == '__main__':
    main()
