# -*- coding: utf-8 -*-
"""探针 A 全量 100 张渲染 (2026-09-12, 修正版)

背景/教训: D:/tmp_timing/render_test100.py 的 BASE 硬编码为 VOSR_1.4B_os (dim1536),
而 v7 LoRA 属 0.5B 家族 (dim1024) -> 每个 case 都 "size mismatch for mlp_ca.fc2.lora_B"
-> 静默 [SKIP] 空转 36 分钟 0 出图。本脚本按 --arch 正确选择基座与推理脚本。

复用: probe_a/pred_sc0.4 已有 30 张 sc0.4 PNG (16:25-16:54 单进程渲染).
由于 vosr_lora_infer.py 在循环外 torch.manual_seed(seed) 且输入目录按名排序,
该 30 张 == 全量 100 张排序后的前缀, 逐字节等价 -> 直接复用, 省 ~29 分钟。

用法: python render_probe_a_100.py --scale 0.4 --tag probeA_sc04 [--no-reuse]
"""
from __future__ import annotations

import argparse
import os
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
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
SEED = 8
REUSE_DIR = Path(r'D:/models/probe_a/pred_sc0.4')      # 已渲染的 30 张 sc0.4 PNG
CWD = r'C:/models/finals'                              # 关键: vosr 内相对路径 preset/ckpts 依赖 cwd

ARCH = {
    '05b': (r'C:/models/finals/preset/ckpts/VOSR_0.5B_os', r'D:/tmp_timing/vosr_lora_infer.py'),
    '14b': (r'C:/models/finals/preset/ckpts/VOSR_1.4B_os', r'D:/tmp_timing/vosr_lora_infer_14b.py'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--arch', choices=list(ARCH), default='05b')
    ap.add_argument('--no-reuse', action='store_true')
    a = ap.parse_args()

    base, infer = ARCH[a.arch]
    root = MODELS / f'final_compliant_{a.tag}'
    png_dir = root / 'render_png'
    out_dir = root / 'output_dir'
    png_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    logp = root / 'render_probe_a_100.log'

    def log(m):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
        print(line, flush=True)
        with logp.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    # 基座 / LoRA 维度一致性前置断言 (防止 1.4B<->0.5B 混用再次静默失败空转)
    import json
    import re
    import torch
    aj = json.loads((Path(base) / 'args.json').read_text(encoding='utf-8'))
    dim, depth = aj['dim'], aj['depth']
    lora = torch.load(LORA, map_location='cpu', weights_only=False)
    lsd = lora['state_dict']
    # 0.5B: mlp_ca.fc2.lora_B = [1024,32], 28 blocks | 1.4B: [1536,32], 36 blocks
    key = [k for k in lsd if k.endswith('mlp_ca.fc2.lora_B.default.weight')][0]
    lora_dim = lsd[key].shape[0]
    lora_depth = len({m.group(1) for k in lsd if (m := re.search(r'blocks\.(\d+)\.', k))})
    log(f'[assert] base={Path(base).name} dim={dim} depth={depth} | '
        f'lora dim={lora_dim} depth={lora_depth} variant={lora.get("variant")} step={lora.get("step")}')
    if lora_dim != dim or lora_depth != depth:
        log(f'   !! 不匹配 -> 中止 (会把每个 case 都 size mismatch 空转)'); return
    log(f'   [ok] {Path(infer).name} + {Path(base).name} 一致')
    del lora, lsd

    files = sorted(TEST.glob('*.jpg'))
    log(f'[boot] 测试集 {len(files)} 张, scale={a.scale}, arch={a.arch}')

    # ① 复用已渲染的 30 张 (仅当非 --no-reuse 且存在于 REUSE_DIR)
    n_reuse = 0
    if not a.no_reuse and REUSE_DIR.exists():
        for p in sorted(REUSE_DIR.glob('*.png')):
            if p.stem in {f.stem for f in files} and not (png_dir / p.name).exists():
                shutil.copy2(p, png_dir / p.name)
                n_reuse += 1
    log(f'① 复用 {n_reuse} 张 -> {png_dir}')

    # ② 剩余图片放独立输入目录, 单进程批处理 (避免每张重复加载模型)
    todo = [f for f in files if not (png_dir / f'{f.stem}.png').exists()]
    log(f'② 待渲染 {len(todo)} 张')
    if todo:
        lq = root / 'lq_rest'
        if lq.exists():
            shutil.rmtree(lq)
        lq.mkdir(parents=True, exist_ok=True)
        for f in todo:
            shutil.copy2(f, lq / f.name)
        rest_png = root / 'render_rest'
        rest_png.mkdir(parents=True, exist_ok=True)
        cmd = [PY, infer, '-c', base, '--lora', LORA, '--lora-scale', str(a.scale),
               '--seed', str(SEED), '--align_method', 'wavelet',
               '-i', str(lq), '-o', str(rest_png)]
        log(f'   渲染中: {" ".join(cmd[1:6])} ... ({len(todo)} 张, 预计 {len(todo)*58/60:.0f} min)')
        t0 = time.time()
        with (root / 'render_rest.out.log').open('w', encoding='utf-8') as fo:
            r = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=CWD)
        got = sorted(rest_png.glob('*.png'))
        log(f'   rc={r.returncode}, 出图 {len(got)} 张, 耗时 {(time.time()-t0)/60:.1f} min')
        for p in got:
            shutil.move(str(p), str(png_dir / p.name))

    have = sorted(png_dir.glob('*.png'))
    log(f'③ PNG 合计 {len(have)}/100')
    if len(have) < 100:
        log('   !! 不足 100 张, 停止'); return

    # ④ PNG -> q95 jpg (与提交管线同规格)
    from PIL import Image
    for p in have:
        j = out_dir / f'{p.stem}.jpg'
        if j.exists():
            continue
        Image.open(p).convert('RGB').save(j, quality=95)
    n_jpg = len(list(out_dir.glob('*.jpg')))
    log(f'④ 转换完成: {n_jpg}/100 jpg -> {out_dir}')
    print('RENDER_PROBE_A_DONE' if n_jpg == 100 else f'PARTIAL {n_jpg}/100')


if __name__ == '__main__':
    main()
