# -*- coding: utf-8 -*-
"""sc0.0 收尾后转速度轴 (2026-09-12 夜, 用户指示: 不着急 sc1.0)
================================================================
背景: 探针 A 已提交但官方队列很久 -> 短期拿不到分数反馈,
      不能"等分再动"; 且 D3(加速比 >1) 是四维里最大的零覆盖区。
本链在 sc0.0 打包完成后:
  ① 打包 + 10 项门禁 sc0.0  (保真极点, 第 2 发弹药)
  ② speed_breakdown_512.py  (四组件时延分解 + DiT 剪枝收益【实测】)
  ③ probe A 本地 niqe 复核 (frugal 断点续跑版)
【不做 sc1.0】
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
OUTDIR = MODELS / 'final_compliant_M24sc00' / 'output_dir'
LOG = T / 'chain_after_sc00.log'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(script, args, timeout, tag, tail=14):
    log(f'--- {tag} ---')
    with (T / f'after_{tag}.log').open('w', encoding='utf-8') as fo:
        r = subprocess.run([PY, '-u', str(T / script)] + args, stdout=fo,
                           stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout, cwd=r'C:/models/finals')
    txt = (T / f'after_{tag}.log').read_text(encoding='utf-8', errors='replace')
    for ln in txt.splitlines()[-tail:]:
        log('   | ' + ln)
    return r.returncode


def main():
    log('=== sc0.0 收尾 → 速度轴 链启动 (不做 sc1.0) ===')
    t0 = time.time()
    while len(list(OUTDIR.glob('*.jpg'))) < 100:
        if time.time() - t0 > 180 * 60:
            log('   !! 等 sc0.0 出图超时'); return
        time.sleep(60)
    log(f'   sc0.0 出图完成: {len(list(OUTDIR.glob("*.jpg")))}/100')

    # ① 打包 + 门禁 (保真极点)
    run('build_probe_a.py', ['--scale', '0.0', '--tag', 'M24sc00', '--skip-render'],
        120 * 60, 'pack_sc00', tail=16)

    # ② 速度轴: 时延分解 + 剪枝收益实测 (安静 GPU)
    run('speed_breakdown_512.py', [], 3600, 'speed_breakdown', tail=30)

    # ③ 探针 A 本地 niqe 复核 (断点续跑, 省内存版)
    run('verify_probe_a_frugal.py', ['--n', '30'], 3 * 3600, 'probeA_niqe', tail=14)

    log('=== 链结束; upload/ 现状 ===')
    for p in sorted((MODELS / 'upload').glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('CHAIN_AFTER_SC00_DONE')


if __name__ == '__main__':
    main()
