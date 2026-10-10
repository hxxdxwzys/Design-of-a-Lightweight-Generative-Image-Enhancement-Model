# -*- coding: utf-8 -*-
"""B 测试集直选 (2026-08-19): 渲染 3 个代表 ckpt 到测试集 -> 入池 -> choice -> 汇总唤醒
代表 ckpt: v1_s2500(保守) / v2_s1500(均衡) / v3_s500(激进), align=wavelet 对齐池内 VOSR 语义
用法: D:/Python/python.exe b_final.py
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
LOG = TT / 'b_final.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
RENDER_OUT = Path(r'D:/models/sweep_test_out')

JOBS = [
    ('v1', 'lora_s2500.pt'),
    ('v2', 'lora_s1500.pt'),
    ('v3', 'lora_s500.pt'),
]
# 渲染输出必须与池目录隔离! pool_add_source 会 rmtree 目标目录 (同名即互删)
RENDER_BASE = OUT


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
    if r.returncode != 0:
        tail = logfile.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-8:]
        for t in tail:
            log(f'  | {t}')
    return r.returncode == 0


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('[boot] B 测试集直选启动 (v2: 渲染与池目录隔离)')
    for variant, ck in JOBS:
        tag = f'{variant}_{ck[:-3]}'
        test_out = RENDER_BASE / variant / f'test_out_{tag}'
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
        log(f'[pool] {tag}')
        ok = run([PY, str(TT / 'pool_add_source.py'), '--src', str(test_out), '--name', f'vosr_lora_{tag}'],
                 TT, TT / f'b_final_pool_{tag}.log', f'pool_{tag}')
        if not ok:
            log(f'[err] {tag} 入池失败 -> 唤醒代理')
            sys.exit(3)
    run([PY, str(TT / 'compute_choice.py')], TT, TT / 'b_final_choice.log', 'choice_final')
    extra = []
    try:
        sr = json.loads(Path(r'D:/models/sweep_test_out/sweep_results.json').read_text(encoding='utf-8'))
        for r in sr['results']:
            if r['name'].startswith('vosr_lora'):
                extra.append(f"{r['name']}: test_clipiqa={r['test_clipiqa']}")
        cm = json.loads(Path(r'D:/tmp_timing/choice_maps.json').read_text(encoding='utf-8'))
        for m in ('M0.000', 'M0.005', 'M0.010'):
            mm = cm['maps'].get(m)
            if mm:
                extra.append(f'{m}: mean={mm["mean"]} delta={mm["delta"]} swaps={mm["n_swap"]}')
    except Exception as e:
        extra.append(f'[warn] 汇总失败: {e}')
    (TT / 'b_final_summary.txt').write_text('\n'.join(['=== B 测试集直选汇总 ===',
        f'时间: {time.strftime("%Y-%m-%d %H:%M:%S")}'] + extra) + '\n', encoding='utf-8')
    log('[done] B 测试集直选完成, 唤醒代理')
    sys.exit(0)


if __name__ == '__main__':
    main()
