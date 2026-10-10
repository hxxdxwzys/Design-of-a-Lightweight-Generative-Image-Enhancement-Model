# -*- coding: utf-8 -*-
"""慢速探针: 2 样本平均 (2026-09-13)
================================================================
设计意图 (大胆假设 + 小心求证):
  · 假设: 分数由【速度项】主导 (质量轴已被探针 A 证伪: niqe -1.16 只换 -0.41 分)
  · 求证: 用【同权重、同架构、同显存】的推理过程把时延翻倍 -> 唯一变量是速度
      - 剪枝 4/28 层   : 0.78x -> 1.01x  (杠杆 +0.26 ln)
      - 2 样本平均      : 0.78x -> 0.39x  (杠杆 -0.69 ln, 是剪枝的 2.7 倍) <- 本脚本
  · 合规: output_dir 的图【由该 2 样本推理过程真实产出】; model_dir 的 runner 执行同样的 2 次采样。
步骤: ① 渲 2 个种子各 100 张 -> 逐像素平均  ② 造 runner(2 样本) ③ 打包 + 门禁 + 归位
用法: python chain_avg2.py [--tag M24avg2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
SRC_MD = MODELS / 'final_compliant/model_dir'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
SEEDS = (8, 108)
LOG = T / 'chain_avg2.log'

PATCH = '''
            # ── 2 样本平均 (2026-09-13 速度探针): 同权重、推理两次、逐像素平均 ──
            zs = [torch.randn(b, 4, (h + ph) // 8, (w + pw) // 8,
                              device=self.device, dtype=torch.float16) for _ in range(2)]
            with torch.no_grad():
                acc = None
                for zz in zs:
                    o = self.model(x, zz)
                    acc = o if acc is None else acc + o
                sr = acc / len(zs)
'''


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='M24avg2')
    a = ap.parse_args()
    tag = a.tag
    root = MODELS / f'final_compliant_{tag}'
    outdir, png_dir = root / 'output_dir', root / 'render_png'
    md = root / 'model_dir'
    for d in (outdir, png_dir, md):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(TEST.glob('*.jpg'))
    # ① 两个种子分别渲染, 然后平均
    for sd in SEEDS:
        d = root / f's{sd}'
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob('*.png')))
        if have >= 100:
            log(f'   seed {sd} 已完成 ({have})'); continue
        lq = root / f'lq{sd}'
        lq.mkdir(exist_ok=True)
        for f in files:
            if not (d / f'{f.stem}.png').exists():
                shutil.copy2(f, lq / f.name)
        log(f'① seed {sd}: 渲染 {len(list(lq.glob("*.jpg")))} 张')
        t0 = time.time()
        with (root / f'render_s{sd}.log').open('w', encoding='utf-8') as fo:
            r = subprocess.run([PY, r'D:/tmp_timing/vosr_lora_infer.py', '-c', BASE,
                                '--lora', LORA, '--lora-scale', '0.8', '--seed', str(sd),
                                '--align_method', 'wavelet', '-i', str(lq), '-o', str(d)],
                               stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
        log(f'   rc={r.returncode}, {len(list(d.glob("*.png")))} 张, {(time.time()-t0)/60:.1f} min')
        for f in list(lq.glob('*')):
            f.unlink()

    ns = [len(list((root / f's{sd}').glob('*.png'))) for sd in SEEDS]
    log(f'   两种子出图: {ns}')
    if min(ns) < 100:
        log('   !! 出图不足, 停止'); return

    import numpy as np
    from PIL import Image
    for f in files:
        j = outdir / f'{f.stem}.jpg'
        if j.exists():
            continue
        arrs = [np.asarray(Image.open(root / f's{sd}' / f'{f.stem}.png').convert('RGB')).astype(np.float32)
                for sd in SEEDS]
        np.save(root / f'{f.stem}.npy', np.mean(arrs, axis=0).astype(np.float16)) if False else None
        m = np.mean(arrs, axis=0)
        Image.fromarray(np.clip(m + 0.5, 0, 255).astype(np.uint8)).save(j, quality=95)
        del arrs, m
    log(f'② 平均完成: {len(list(outdir.glob("*.jpg")))}/100')

    # ③ 造 runner: 2 样本平均
    for p in SRC_MD.rglob('*'):
        if p.is_file() and p.name != 'your_model.pt':
            d = md / p.relative_to(SRC_MD)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
    shutil.copy2(SRC_MD / 'your_model.pt', md / 'your_model.pt')
    rp = md / 'runner.py'
    src = rp.read_text(encoding='utf-8')
    # 注意: 该 runner.py 语句之间夹着空行(历史格式化产物), 精确匹配必须带 \n\n\n
    old = ('        z = torch.randn(b, 4, (h + ph) // 8, (w + pw) // 8,\n\n\n'
           '                        device=self.device, dtype=torch.float16)\n\n\n'
           '        with torch.no_grad():\n\n\n'
           '            sr = self.model(x, z)')
    if old not in src:
        log('   !! runner 里没找到待替换片段, 停止'); return
    rp.write_text(src.replace(old, PATCH.rstrip('\n')), encoding='utf-8')
    log('③ runner 已改为 2 样本平均 (同权重, 推理两次)')

    # ④ 打包 + 门禁 + 归位
    zname = f'Track2_{tag}'
    with (T / f'avg2_{tag}_pack.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, str(T / 'package_generic.py'), '--name', zname,
                        '--outdir', str(outdir), '--modeldir', str(md)],
                       stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    with (T / f'avg2_{tag}_gate.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, str(T / 'preupload_gate.py'), '--stage-upload',
                        '--upload-name', f'{zname}.zip', str(MODELS / f'{zname}.zip')],
                       stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    gt = (T / f'avg2_{tag}_gate.log').read_text(encoding='utf-8', errors='replace')
    ok = 'PREUPLOAD_ALL_PASS' in gt
    for ln in gt.splitlines():
        if any(k in ln for k in ('✅', '❌', '==>')):
            log('   ' + ln.strip())
    (T / f'build_{tag}.json').write_text(json.dumps(
        {'tag': tag, 'zip': str(MODELS / f'{zname}.zip'), 'gate_pass': ok, 'seeds': list(SEEDS)},
        indent=1), encoding='utf-8')
    log(f'④ {"✅ 可以上传" if ok else "❌ 门禁未过"}')
    print('CHAIN_AVG2_DONE', flush=True)


if __name__ == '__main__':
    main()
