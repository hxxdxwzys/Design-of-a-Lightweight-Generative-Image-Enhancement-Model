# -*- coding: utf-8 -*-
"""波 2 靶点测试集 100 张 4K 渲染 (2026-09-10 夜)

链条: remerge_scale_14b.py 反推目标 scale -> vosr_pkg_infer_14b.py 渲 100 张
      -> PNG 转 JPG(q95) -> 删临时 pkg 释放磁盘
特性: 断点续跑 (缺哪张渲哪张, 用 hardlink 组 staging 目录避免重复加载模型);
      逐图失败重试后跳过; 全流程写日志。

与生产链的差别: 生产用 vosr_lora_infer_14b.py + LoRA 原件; 本脚本用合并 pkg 直渲。
已证明两者等价 (HANDOFF §16.83: m0s1-sc0.93 实测 lpips 完全一致)。

用法:
  D:/Python/python.exe render_pkg_test100.py --src-pkg <your_model.pt> --scale 0.97 --tag m0s7_s097
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
BASE_C = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
TEST_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
ROOT_BASE = Path(r'D:/models')

# 离线优先: load_dinov2 / from_pretrained 会走 HTTPS 探网 (校园网挡 HF -> WinError 10060 超时,
# 2026-09-11 00:02 实测 m0s7 候选因此整轮失败)。缓存已在本地, 直接强制离线。
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-pkg', required=True)
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keep-pkg', action='store_true', help='保留临时 remerge pkg (默认渲完删掉省 3.5GB)')
    args = ap.parse_args()

    ROOT = ROOT_BASE / f'wave2_{args.tag}'
    PNG_DIR = ROOT / 'render_png'
    OUT_DIR = ROOT / 'output_dir'
    PKG = ROOT / 'model_dir' / 'your_model.pt'
    LOG = ROOT / 'render.log'
    for d in (PNG_DIR, OUT_DIR, ROOT / 'model_dir'):
        d.mkdir(parents=True, exist_ok=True)

    def log(m):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
        print(line, flush=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    jpgs = sorted(TEST_DIR.glob('case*.jpg'))
    log(f'[boot] tag={args.tag} scale={args.scale} 测试集 {len(jpgs)} 张')
    t_all = time.time()

    # ① 反推 pkg (可复用)
    if not PKG.exists():
        r = subprocess.run([PY, str(TOOLS / 'remerge_scale_14b.py'), '--src-pkg', args.src_pkg,
                            '--scale', str(args.scale), '--out', str(PKG)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(f'!! remerge 失败: {r.stdout[-500:]} {r.stderr[-500:]}')
            return
        log(f'[remerge] {PKG} ({PKG.stat().st_size/1e9:.2f} GB)')

    # ② 找出缺的图, 用 hardlink 组 staging 目录 (模型只加载一次)
    #    已完成 = jpg 或 png 任一存在 (png 已在也算, 后面只补转 jpg —— 否则中途被杀
    #    会丢掉已渲的 png, 整批重来; 2026-09-11 实测因此白跑 1.6h)
    def _have(j):
        return (OUT_DIR / f'{j.stem}.jpg').exists() or (PNG_DIR / f'{j.stem}.png').exists()

    missing = [j for j in jpgs if not _have(j)]
    done = len(jpgs) - len(missing)
    log(f'[resume] 已完成 {done}/{len(jpgs)} (jpg 或 png 在), 待渲 {len(missing)}')
    if missing:
        STAGE = ROOT / 'stage_in'
        if STAGE.exists():
            shutil.rmtree(STAGE)
        STAGE.mkdir(parents=True)
        for j in missing:
            try:
                os.link(j, STAGE / j.name)
            except OSError:
                shutil.copy2(j, STAGE / j.name)
        log(f'[stage] {len(missing)} 张 -> {STAGE}')
        t1 = time.time()
        r = subprocess.run([PY, str(TOOLS / 'vosr_pkg_infer_14b.py'), '-c', BASE_C,
                            '--pkg', str(PKG), '--fp16', '-i', str(STAGE), '-o', str(PNG_DIR),
                            '--align_method', 'wavelet'],
                           capture_output=True, text=True, timeout=8 * 3600,
                           cwd=r'C:/models/finals')
        tail = (r.stdout or '')[-1500:] + (r.stderr or '')[-1500:]
        log(f'[render] rc={r.returncode} 耗时 {(time.time()-t1)/60:.1f} min\n{tail.strip()}')
        if r.returncode != 0:
            log('!! 渲染进程非零退出, 继续尝试已产出的 png 转 jpg')

    # ③ PNG -> JPG q95 (缺失的)
    from PIL import Image
    for j in jpgs:
        out_jpg = OUT_DIR / f'{j.stem}.jpg'
        if out_jpg.exists():
            continue
        png = PNG_DIR / f'{j.stem}.png'
        if not png.exists():
            log(f'[skip] {j.stem}: 无 png')
            continue
        img = Image.open(png).convert('RGB')
        img.save(out_jpg, quality=95)
        done += 1
        log(f'[jpg] {j.stem} ({done}/{len(jpgs)})')

    log(f'[done] {args.tag}: {done}/{len(jpgs)} 张, 总耗时 {(time.time()-t_all)/60:.1f} min')
    if done == len(jpgs):
        if not args.keep_pkg:
            if PKG.exists():
                PKG.unlink()
                log('[cleanup] 已删临时 pkg')
        log('RENDER_PKG_DONE')
    else:
        log(f'RENDER_PKG_PARTIAL {done}/{len(jpgs)}')


if __name__ == '__main__':
    main()
