# -*- coding: utf-8 -*-
"""B 攻击收尾 (2026-08-20): 渲染 v5_s3000/s4000 测试集 -> 入池 -> choice -> 汇总唤醒
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
LOG = TT / 'b_attack3.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'

CKS = ['v5_s3000', 'v5_s4000']


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('[boot] B 攻击收尾启动')
    for ck in CKS:
        test_out = OUT / 'v5' / f'test_out_{ck}'
        if test_out.exists() and len(list(test_out.glob('*.png'))) >= 100:
            log(f'[skip] {ck} 渲染已存在')
        else:
            test_out.mkdir(parents=True, exist_ok=True)
            log(f'[render] {ck} 测试集 100 张 (~1.6h, wavelet)')
            with (OUT / 'v5' / 'test_render.log').open('ab') as f:
                r = subprocess.run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT,
                                    '--lora', str(OUT / 'v5' / f'{ck}.pt'),
                                    '-i', TEST_DIR, '-o', str(test_out),
                                    '--align_method', 'wavelet', '--seed', '42'],
                                   cwd=str(OUT / 'v5'), stdout=f, stderr=subprocess.STDOUT)
            log(f'[render] {ck} exit={r.returncode}')
            if r.returncode != 0 or len(list(test_out.glob('*.png'))) < 100:
                log(f'[err] {ck} 渲染失败 -> 唤醒代理')
                sys.exit(2)
        with (TT / f'b_attack3_pool_{ck}.log').open('ab') as f:
            subprocess.run([PY, str(TT / 'pool_add_source.py'), '--src', str(test_out),
                            '--name', f'vosr_lora_{ck}'], cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
        log(f'[pool] {ck} 完成')
    with (TT / 'b_attack3_choice.log').open('ab') as f:
        subprocess.run([PY, str(TT / 'compute_choice.py')], cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
    extra = []
    try:
        sr = json.loads(Path(r'D:/models/sweep_test_out/sweep_results.json').read_text(encoding='utf-8'))
        for r in sr['results']:
            if r['name'].startswith('vosr_lora_v5'):
                extra.append(f"{r['name']}: {r['test_clipiqa']}")
        cm = json.loads(Path(r'D:/tmp_timing/choice_maps.json').read_text(encoding='utf-8'))
        for m in ('M0.000', 'M0.010'):
            mm = cm['maps'].get(m)
            if mm:
                extra.append(f'{m}: mean={mm["mean"]} swaps={mm["n_swap"]}')
    except Exception as e:
        extra.append(f'[warn] {e}')
    (TT / 'b_attack3_summary.txt').write_text('\n'.join(['=== B 攻击收尾汇总 ===', time.strftime('%Y-%m-%d %H:%M:%S')] + extra) + '\n', encoding='utf-8')
    log('[done] 攻击收尾完成, 唤醒代理')
    sys.exit(0)


if __name__ == '__main__':
    main()
