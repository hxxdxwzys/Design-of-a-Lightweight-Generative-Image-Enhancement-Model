# -*- coding: utf-8 -*-
"""B 攻击链 (2026-08-20): 渲染 v3_s1500/v3_s2500/v2_s2500 入池 -> choice -> v5 激进训练(w_clip=2.5, 4000步) -> 门控 -> 唤醒
用法: D:/Python/python.exe b_attack.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
LOG = TT / 'b_attack.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
VAL_IN = TT / 'codiff_val_in'

RENDERS = [
    ('v3', 'lora_s1500.pt'),
    ('v3', 'lora_s2500.pt'),
    ('v2', 'lora_s2500.pt'),
]


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, cwd, logfile, tag):
    log(f'[run] {tag}')
    t0 = time.time()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('ab') as f:
        r = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    log(f'[run] {tag}: exit={r.returncode} ({time.time() - t0:.0f}s)')
    return r.returncode == 0


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('[boot] B 攻击链启动')
    # 1) 渲染 3 个高 clip 档入池
    for variant, ck in RENDERS:
        tag = f'{variant}_{ck[:-3]}'
        test_out = OUT / variant / f'test_out_{tag}'
        if test_out.exists() and len(list(test_out.glob('*.png'))) >= 100:
            log(f'[skip] {tag} 渲染已存在')
        else:
            test_out.mkdir(parents=True, exist_ok=True)
            log(f'[render] {tag} 测试集 100 张 (~1.6h, wavelet)')
            ok = run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / variant / ck),
                      '-i', TEST_DIR, '-o', str(test_out), '--align_method', 'wavelet', '--seed', '42'],
                     TT, OUT / variant / 'test_render.log', f'test_render_{tag}')
            if not ok or len(list(test_out.glob('*.png'))) < 100:
                log(f'[err] {tag} 渲染失败 -> 唤醒代理')
                sys.exit(2)
        run([PY, str(TT / 'pool_add_source.py'), '--src', str(test_out), '--name', f'vosr_lora_{tag}'],
            TT, TT / f'b_attack_pool_{tag}.log', f'pool_{tag}')
    run([PY, str(TT / 'compute_choice.py')], TT, TT / 'b_attack_choice.log', 'choice_after_renders')

    # 2) v5 激进训练 (w_clip=2.5, 4000 步)
    log('[train] v5 (w_clip=2.5, 4000 步, 每 1000 存盘)')
    run([PY, str(TT / 'train_vosr_lora.py'), '--variant', 'v5', '--w-clip', '2.5',
         '--w-lpips', '1.0', '--w-l1', '0.05', '--steps', '4000', '--save-every', '1000',
         '--lr', '3e-5', '--ckpt-prefix', 'v5'],
        TT, OUT / 'v5' / 'train.log', 'train_v5')

    # 3) v5 val 门控
    for ck in sorted((OUT / 'v5').glob('v5_s*.pt')):
        tag = f'v5_{ck.stem}'
        val_out = OUT / 'v5' / f'val_out_{tag}'
        gj = OUT / 'v5' / f'val_gate_{tag}.json'
        log(f'[valgate] {tag}')
        run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(ck),
             '-i', str(VAL_IN), '-o', str(val_out), '--align_method', 'nofix', '--seed', '42'],
            OUT / 'v5', OUT / 'v5' / 'val_render.log', f'val_render_{tag}')
        run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
            TT, TT / 'b_attack_valgate.log', f'val_gate_{tag}')
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        log(f'[valgate] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")}')
    log('[done] B 攻击链完成, 唤醒代理')
    sys.exit(0)


if __name__ == '__main__':
    main()
