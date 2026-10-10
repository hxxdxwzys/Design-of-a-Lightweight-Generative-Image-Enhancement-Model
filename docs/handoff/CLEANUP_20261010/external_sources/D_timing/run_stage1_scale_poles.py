# -*- coding: utf-8 -*-
"""阶段 1 剂量-反应链 (2026-09-12): 探针 A 打包完成后, 依次渲染 + 打包 scale 轴的两个极点。

设计: 同一模型/同一 runner/同一速度, 只有 LoRA scale 不同 -> **唯一变量是图像质量**。
四点曲线 (sc0.0 / sc0.4 / sc0.8[已完成, 官方 43.6951] / sc1.0) 的 argmax 直接判别得分轴:
  niqe 主导   -> 峰值在 sc0.4 (val niqe 4.459 最低)
  lpips/dists -> 峰值在 sc0.0 (val lpips 0.342 / dists 0.2105 最低)
  musiq/maniqa-> 峰值在 sc0.8 (val musiq 48.14 / maniqa 0.514 最高)
  clipiqa     -> 峰值在 sc1.0 (val clipiqa 0.938 最高)
每个点 100 张 4K ≈ 95 min 渲染 + 打包门禁。

用法: python run_stage1_scale_poles.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
UPLOAD = MODELS / 'upload'
LOG = T / 'stage1_scale_poles.log'

# (tag, scale, 说明)
POINTS = [
    ('M24sc00', 0.0, '保真极点 (纯基座 VOSR, 无 LoRA)'),
    ('M24sc10', 1.0, '感知极点 (LoRA 满强度)'),
]

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(args, timeout, tag):
    with (T / f'stage1_{tag}.log').open('w', encoding='utf-8') as fo:
        r = subprocess.run([PY, '-u'] + args, stdout=fo, stderr=subprocess.STDOUT,
                           text=True, encoding='utf-8', errors='replace',
                           timeout=timeout, cwd=r'C:/models/finals')
    tail = (T / f'stage1_{tag}.log').read_text(encoding='utf-8', errors='replace').splitlines()[-6:]
    return r.returncode, tail


def main():
    log('=== 阶段 1 剂量-反应链启动 ===')
    marker = UPLOAD / 'Track2_probeA_sc04.zip'
    t0 = time.time()
    while not marker.exists():
        if time.time() - t0 > 240 * 60:
            log('   !! 等待探针 A 归位超时 (4h), 仍继续'); break
        time.sleep(60)
    log(f'   探针 A 已归位: {marker.name} ({marker.stat().st_size/2**30:.2f} GiB)')

    for tag, scale, desc in POINTS:
        log(f'--- {tag} (scale={scale}) {desc} ---')
        rc, tail = run([str(T / 'render_probe_a_100.py'), '--scale', str(scale),
                        '--tag', tag, '--no-reuse'], 200 * 60, f'{tag}_render')
        for ln in tail:
            log('   | ' + ln)
        n = len(list((MODELS / f'final_compliant_{tag}' / 'output_dir').glob('*.jpg')))
        log(f'   渲染 rc={rc}, 出图 {n}/100')
        if n < 100:
            log('   !! 出图不足, 跳过该点'); continue
        rc, tail = run([str(T / 'build_probe_a.py'), '--scale', str(scale),
                        '--tag', tag, '--skip-render'], 120 * 60, f'{tag}_build')
        for ln in tail:
            log('   | ' + ln)
        bj = T / f'build_{tag}.json'
        if bj.exists():
            log('   ' + bj.read_text(encoding='utf-8').replace('\n', ' '))

    log('=== 阶段 1 结束: upload/ 现状 ===')
    for p in sorted(UPLOAD.glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('STAGE1_SCALE_POLES_DONE')


if __name__ == '__main__':
    main()
