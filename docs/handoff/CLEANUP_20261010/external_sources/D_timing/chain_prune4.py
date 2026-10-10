# -*- coding: utf-8 -*-
"""剪枝速度探针全自动链 (2026-09-13)
================================================================
依据 (实测, 非估算):
  · 出货版 512 全链 364.5 ms -> 比基线 0.784x;  DiT 28 层占 99.0 ms (28%)
  · DiT 截断实测: 14层 322ms(0.89x) / 7层 291ms(0.98x) / 4层 282ms(1.01x)
  · VAE encode(111.8) + LightDecoder(124.3) = 65% 不可剪 (换掉需重训) => 1.01x 是今晚的提速上限
目标: 作出【真加速】且合规的包 —— 图由剪枝后的同一模型产出, model_dir 与之一一对应。
步骤: ① 造剪枝 ckpt  ② 用它渲 100 张 4K  ③ 造对应 pkg   ④ 打包 + 10 项门禁 + 归位
用法: python chain_prune4.py [--keep 4] [--tag M24p4]
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
SRC_CKPT = Path(r'C:/models/finals/preset/ckpts/VOSR_0.5B_os')
SRC_MD = MODELS / 'final_compliant/model_dir'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
SCALE = 0.8
LOG = T / 'chain_prune4.log'


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', type=int, default=4)
    ap.add_argument('--tag', default='M24p4')
    a = ap.parse_args()
    tag, keep = a.tag, a.keep
    # 必须放在 preset/ckpts/ 下: vosr_lora_infer.py 按 dirname(ckpt)/stable-diffusion-2-1-base 找 VAE
    ckpt = Path(r'C:/models/finals/preset/ckpts') / f'VOSR_0.5B_{tag}'
    root = MODELS / f'final_compliant_{tag}'
    outdir, png_dir = root / 'output_dir', root / 'render_png'
    md = root / 'model_dir'
    for d in (ckpt, outdir, png_dir, md):
        d.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import load_file, save_file

    # ① 剪枝 ckpt (保留 blocks 0..keep-1, args.json depth=keep)
    log(f'① 造剪枝 ckpt: keep={keep}/28 -> {ckpt}')
    args_j = json.loads((SRC_CKPT / 'args.json').read_text(encoding='utf-8'))
    src_depth = args_j['depth']
    args_j['depth'] = keep
    (ckpt / 'args.json').write_text(json.dumps(args_j, indent=4), encoding='utf-8')
    (ckpt / 'checkpoints').mkdir(exist_ok=True)
    sd = load_file(str(SRC_CKPT / 'checkpoints' / 'ema_model.safetensors'))
    kept, dropped = {}, 0
    for k, v in sd.items():
        if k.startswith('blocks.'):
            n = int(k.split('.')[1])
            if n >= keep:
                dropped += 1
                continue
        kept[k] = v
    save_file(kept, str(ckpt / 'checkpoints' / 'ema_model.safetensors'))
    log(f'   {len(sd)} -> {len(kept)} 键 (丢弃 {dropped}) | depth {src_depth} -> {keep}')

    # ② 渲染 100 张 4K (用剪枝 ckpt + 同 LoRA 同 scale)
    todo = [f for f in sorted((Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')).glob('*.jpg'))
            if not (png_dir / f'{f.stem}.png').exists()]
    log(f'② 渲染 {len(todo)} 张 (scale={SCALE})')
    if todo:
        lq = root / 'lq_rest'
        lq.mkdir(exist_ok=True)
        for f in todo:
            shutil.copy2(f, lq / f.name)
        t0 = time.time()
        with (root / 'render.out.log').open('w', encoding='utf-8') as fo:
            r = subprocess.run([PY, r'D:/tmp_timing/vosr_lora_infer.py', '-c', str(ckpt),
                                '--lora', LORA, '--lora-scale', str(SCALE), '--seed', '8',
                                '--align_method', 'wavelet', '-i', str(lq), '-o', str(png_dir)],
                               stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
        log(f'   rc={r.returncode}, 出图 {len(list(png_dir.glob("*.png")))} 张, {(time.time()-t0)/60:.1f} min')
    n_png = len(list(png_dir.glob('*.png')))
    if n_png < 100:
        log(f'   !! 出图不足 {n_png}/100, 停止'); return
    from PIL import Image
    for p in sorted(png_dir.glob('*.png')):
        j = outdir / f'{p.stem}.jpg'
        if not j.exists():
            Image.open(p).convert('RGB').save(j, quality=95)
    log(f'   转 jpg: {len(list(outdir.glob("*.jpg")))}/100')

    # ③ 造对应 pkg (从原 pkg 过滤 DiT 层, 其余键原样)
    log('③ 造 pkg (原 pkg 过滤 DiT 层)')
    pkg = torch.load(str(SRC_MD / 'your_model.pt'), map_location='cpu', weights_only=False, mmap=True)
    diT = {k: v for k, v in pkg['diT'].items()
           if not (k.startswith('blocks.') and int(k.split('.')[1]) >= keep)}
    newpkg = dict(pkg)
    newpkg['diT'] = diT
    newpkg['diT_cfg'] = dict(pkg['diT_cfg'])
    newpkg['diT_cfg']['depth'] = keep
    for p in SRC_MD.rglob('*'):
        if p.is_file() and p.name != 'your_model.pt':
            d = md / p.relative_to(SRC_MD)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
    torch.save(newpkg, str(md / 'your_model.pt'))
    log(f'   diT 键 {len(pkg["diT"])} -> {len(diT)}; depth={keep}; '
        f'model_dir 文件 {sum(1 for x in md.rglob("*") if x.is_file())}')

    # ④ 打包 + 门禁 + 归位
    zname = f'Track2_{tag}'
    log(f'④ 打包 {zname}')
    with (T / f'prune_{tag}_pack.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, str(T / 'package_generic.py'), '--name', zname,
                        '--outdir', str(outdir), '--modeldir', str(md)],
                       stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    log('   ' + (T / f'prune_{tag}_pack.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()[-1])
    with (T / f'prune_{tag}_gate.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, str(T / 'preupload_gate.py'), '--stage-upload',
                        '--upload-name', f'{zname}.zip', str(MODELS / f'{zname}.zip')],
                       stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    gt = (T / f'prune_{tag}_gate.log').read_text(encoding='utf-8', errors='replace')
    ok = 'PREUPLOAD_ALL_PASS' in gt
    for ln in gt.splitlines():
        if any(k in ln for k in ('✅', '❌', '==>')):
            log('   ' + ln.strip())
    (T / f'build_{tag}.json').write_text(json.dumps(
        {'tag': tag, 'keep': keep, 'zip': str(MODELS / f'{zname}.zip'), 'gate_pass': ok},
        indent=1), encoding='utf-8')
    log(f'④ {"✅ 可以上传" if ok else "❌ 门禁未过"}')
    print('CHAIN_PRUNE_DONE', flush=True)


if __name__ == '__main__':
    main()
