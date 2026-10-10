# -*- coding: utf-8 -*-
"""波 2 自愈补跑 (2026-09-11 夜): 等主队列结束后, 把未满 100 张的候选全部补渲 + 补 clipiqa。

存在理由: 2026-09-11 00:02 m0s7 候选因 load_dinov2 走 HTTPS 探网超时整轮失败 (WinError 10060)。
主队列不会回头重试, 故挂本任务在后面兜底 —— 失败/中断的候选自动重跑, 不需要人工介入。

用法: D:/Python/python.exe retry_wave2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
MAIN_LOG = Path(r'D:/tmp_timing/wave2_overnight.log')
OUT = Path(r'D:/tmp_timing/wave2_overnight.json')

SRC_M0S1 = r'D:/models/final_compliant_14b/model_dir/your_model.pt'
SRC_M0S7 = r'D:/models/final_compliant_m0s7/model_dir/your_model.pt'
CANDIDATES = [
    ('m0s7_s097', SRC_M0S7, 0.97, (0.9203, 0.3542), 'T_high'),
    ('m0s1_s090', SRC_M0S1, 0.90, (0.9209, 0.3556), 'T_high'),
    ('m0s1_s097', SRC_M0S1, 0.97, (0.9311, 0.3595), 'T_low'),
]
LOG = Path(r'D:/tmp_timing/wave2_retry.log')


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def renderer_alive():
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                            "Where-Object { $_.CommandLine -match 'run_wave2_overnight|vosr_pkg_infer' } | "
                            "Measure-Object).Count"],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout or '0').strip() or 0) > 0
    except Exception:
        return True


def main():
    log('=== 自愈补跑启动, 等待主队列结束 ===')
    t0 = time.time()
    while time.time() - t0 < 16 * 3600:
        done_marker = MAIN_LOG.exists() and 'WAVE2_OVERNIGHT_DONE' in MAIN_LOG.read_text(
            encoding='utf-8', errors='replace')
        if done_marker and not renderer_alive():
            log('主队列已结束')
            break
        time.sleep(120)
    else:
        log('!! 等待超时 16h, 仍继续执行补跑')

    for tag, src, scale, val_expect, branch in CANDIDATES:
        out_dir = Path(rf'D:/models/wave2_{tag}/output_dir')
        n = len(list(out_dir.glob('*.jpg'))) if out_dir.exists() else 0
        if n >= 100:
            log(f'[skip] {tag}: 已满 {n}/100')
            continue
        log(f'[补跑] {tag}: 现有 {n}/100 -> 重渲')
        r = subprocess.run([PY, str(TOOLS / 'render_pkg_test100.py'), '--src-pkg', src,
                            '--scale', str(scale), '--tag', tag],
                           capture_output=True, text=True, timeout=14 * 3600,
                           env={**os.environ})
        log(f'[render] {tag} rc={r.returncode} :: {(r.stdout or "")[-200:].strip()}')

    # 统一补 clipiqa
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    for tag, src, scale, val_expect, branch in CANDIDATES:
        out_dir = Path(rf'D:/models/wave2_{tag}/output_dir')
        n = len(list(out_dir.glob('*.jpg'))) if out_dir.exists() else 0
        if n == 0:
            continue
        cj = Path(rf'D:/tmp_timing/wave2_clip_{tag}.json')
        if cj.exists():
            c = json.loads(cj.read_text(encoding='utf-8'))
        else:
            r = subprocess.run([PY, str(TOOLS / 'nr_clipiqa_dir.py'), '--dir', str(out_dir),
                                '--out', str(cj)], capture_output=True, text=True, timeout=6 * 3600)
            log(f'[clipiqa] {tag} rc={r.returncode} :: {(r.stdout or "")[-200:].strip()}')
            c = json.loads(cj.read_text(encoding='utf-8')) if cj.exists() else {}
        e = {'branch': branch, 'val_expect': list(val_expect), 'n_jpg': n, 'scale': scale,
             'src_pkg': Path(src).parent.parent.name}
        if c:
            e['test_clip'] = {'n': c.get('n'), 'mean': c.get('mean'), 'std': c.get('std'),
                              'min': c.get('min'), 'max': c.get('max')}
        res[tag] = e
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')

    log('=== 最终汇总 ===')
    for tag, e in res.items():
        tc = e.get('test_clip')
        if tc:
            log(f'  {tag:12s} 分支={e["branch"]:7s} val={e["val_expect"]} '
                f'test_clip={tc["mean"]:.4f} (std {tc["std"]:.4f} min {tc["min"]:.4f}) n={e["n_jpg"]}')
        else:
            log(f'  {tag:12s} {e}')
    log('WAVE2_RETRY_DONE')


if __name__ == '__main__':
    main()
