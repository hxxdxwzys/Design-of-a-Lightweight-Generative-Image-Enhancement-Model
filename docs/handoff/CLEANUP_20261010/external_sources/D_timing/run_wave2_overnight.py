# -*- coding: utf-8 -*-
"""波 2 整夜队列: 依次渲染三个候选靶点 + 算测试集 100 张 clipiqa (2026-09-10 夜)

候选按分支优先级排序 (等官方分判 T 后择一提交):
  1. m0s7-sc0.97   T 高分支首选 (val lpips 0.3542, 全场最优)
  2. m0s1-sc0.90   T 高分支次选 (val 0.9209/0.3556)
  3. m0s1-sc0.97   T 低分支首选 (val clip 0.9311, 本地最高 clip 档)

每个候选: 渲 100 张测试集 4K -> 转 jpg -> 算 clipiqa(100张直证) -> 写汇总。
测试集直证的目的: §16.82 指出 1.4B 的 val->test clipiqa 跌幅达 0.044-0.064,
按 val 选档有踩空风险 -> 直接用真实测试集 clipiqa 做最终裁决。

用法: D:/Python/python.exe run_wave2_overnight.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
OUT = Path(r'D:/tmp_timing/wave2_overnight.json')

SRC_M0S1 = r'D:/models/final_compliant_14b/model_dir/your_model.pt'
SRC_M0S7 = r'D:/models/final_compliant_m0s7/model_dir/your_model.pt'

# (tag, 源 pkg, 目标 scale, 预登记 val (clip, lpips), 适用分支)
CANDIDATES = [
    ('m0s7_s097', SRC_M0S7, 0.97, (0.9203, 0.3542), 'T_high'),
    ('m0s1_s090', SRC_M0S1, 0.90, (0.9209, 0.3556), 'T_high'),
    ('m0s1_s097', SRC_M0S1, 0.97, (0.9311, 0.3595), 'T_low'),
]


def main():
    LOG = Path(r'D:/tmp_timing/wave2_overnight.log')

    def log(m):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
        print(line, flush=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    log(f'=== 波 2 整夜队列启动, {len(CANDIDATES)} 个候选 ===')
    for tag, src, scale, val_expect, branch in CANDIDATES:
        log('=' * 70)
        log(f'候选 {tag} (分支 {branch}, val 预登记 {val_expect})')
        r = subprocess.run([PY, str(TOOLS / 'render_pkg_test100.py'), '--src-pkg', src,
                            '--scale', str(scale), '--tag', tag],
                           capture_output=True, text=True, timeout=12 * 3600)
        log(f'[render] rc={r.returncode} :: {(r.stdout or "")[-300:].strip()}')
        if r.returncode != 0:
            log(f'!! 渲染失败 {(r.stderr or "")[-400:]}')

        out_dir = Path(rf'D:/models/wave2_{tag}/output_dir')
        n_jpg = len(list(out_dir.glob('*.jpg'))) if out_dir.exists() else 0
        log(f'[产物] {out_dir} -> {n_jpg}/100 jpg')
        if n_jpg == 0:
            res[tag] = {'error': 'no output', 'rc': r.returncode}
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
            continue

        clip_json = Path(rf'D:/tmp_timing/wave2_clip_{tag}.json')
        r2 = subprocess.run([PY, str(TOOLS / 'nr_clipiqa_dir.py'), '--dir', str(out_dir),
                             '--out', str(clip_json)], capture_output=True, text=True, timeout=4 * 3600)
        log(f'[clipiqa] rc={r2.returncode} :: {(r2.stdout or "")[-200:].strip()}')
        entry = {'branch': branch, 'val_expect': list(val_expect), 'n_jpg': n_jpg,
                 'src_pkg': Path(src).parent.parent.name, 'scale': scale}
        if clip_json.exists():
            c = json.loads(clip_json.read_text(encoding='utf-8'))
            entry['test_clip'] = {'n': c.get('n'), 'mean': c.get('mean'), 'std': c.get('std'),
                                  'min': c.get('min'), 'max': c.get('max')}
        res[tag] = entry
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        log(f'[记录] {tag}: {json.dumps(entry, ensure_ascii=False)[:300]}')

    log('=' * 70)
    log('汇总 (测试集 100 张 clipiqa 直证):')
    for tag, e in res.items():
        if 'test_clip' in e:
            tc = e['test_clip']
            log(f'  {tag:12s} 分支={e["branch"]:7s} val={e["val_expect"]} '
                f'-> test_clip={tc["mean"]:.4f} (std {tc["std"]:.4f}, min {tc["min"]:.4f})')
        else:
            log(f'  {tag:12s} {e}')
    log('WAVE2_OVERNIGHT_DONE')


if __name__ == '__main__':
    main()
