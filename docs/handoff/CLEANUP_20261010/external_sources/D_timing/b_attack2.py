# -*- coding: utf-8 -*-
"""B 攻击接力 (2026-08-20): 等 b_attack 完成 -> 检查 v5 val 门控
-> 若最佳 v5 clip>=0.93: 自动渲染测试集+入池+choice; 否则只写提示 -> 唤醒
用法: D:/Python/python.exe b_attack2.py
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
LOG = TT / 'b_attack2.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'


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
    log('[boot] 等待 b_attack 完成...')
    t0 = time.time()
    attack_log = TT / 'b_attack.log'
    while time.time() - t0 < 10 * 3600:
        if attack_log.exists() and '[done] B 攻击链完成' in attack_log.read_text(encoding='utf-8', errors='ignore'):
            break
        time.sleep(60)
    log('[step1] b_attack 完成, 检查 v5 门控')
    best = None
    for gj in sorted((OUT / 'v5').glob('val_gate_v5_s*.json')):
        gate = json.loads(gj.read_text(encoding='utf-8'))
        ck = gj.stem.replace('val_gate_', '')  # v5_s1000
        log(f'[gate] {ck}: clip={gate.get("clip")} lpips={gate.get("lpips")}')
        if best is None or gate.get('clip', 0) > best[1]:
            best = (ck, gate.get('clip', 0))
    if best and best[1] >= 0.93:
        ck, clip = best
        ckpt = OUT / 'v5' / f'{ck}.pt'
        test_out = OUT / 'v5' / f'test_out_{ck}'
        log(f'[render] {ck} (val clip={clip} >= 0.93) 测试集 100 张')
        with (OUT / 'v5' / 'test_render.log').open('ab') as f:
            r = subprocess.run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(ckpt),
                                '-i', TEST_DIR, '-o', str(test_out), '--align_method', 'wavelet', '--seed', '42'],
                               cwd=str(OUT / 'v5'), stdout=f, stderr=subprocess.STDOUT)
        log(f'[render] exit={r.returncode}')
        if r.returncode == 0 and len(list(test_out.glob('*.png'))) >= 100:
            with (TT / 'b_attack2_pool.log').open('ab') as f:
                subprocess.run([PY, str(TT / 'pool_add_source.py'), '--src', str(test_out),
                                '--name', f'vosr_lora_{ck}'], cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
            with (TT / 'b_attack2_choice.log').open('ab') as f:
                subprocess.run([PY, str(TT / 'compute_choice.py')], cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
            log('[done] v5 已入池+choice, 唤醒代理')
            sys.exit(0)
    else:
        log(f'[skip] v5 最佳 val clip={best[1] if best else "?"} < 0.93, 不渲染测试集')
    log('[done] 攻击接力完成, 唤醒代理')
    sys.exit(0)


if __name__ == '__main__':
    main()
