# -*- coding: utf-8 -*-
"""结构对比: 现在待交的 5 个包 vs 老世界成功提交过的归档 (2026-09-11)"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NOW = sorted(Path(r'D:/models/upload').glob('*.zip'))
OLD = sorted(Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史').glob('*.zip'))
CASE_RE = re.compile(r'^output_dir/case(\d+)\.jpg$')


def info(p: Path):
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    top = {}
    for n in names:
        top.setdefault(n.split('/')[0], 0)
        top[n.split('/')[0]] += 1
    out = [n for n in names if n.startswith('output_dir/')]
    mds = [n for n in names if n.startswith('model_dir/')]
    cases = sorted(int(CASE_RE.match(n).group(1)) for n in out if CASE_RE.match(n))
    exts = {}
    for n in mds:
        e = Path(n).suffix or '(noext)'
        exts[e] = exts.get(e, 0) + 1
    return {
        'top': top,
        'out_n': len(out),
        'case_ok': cases == list(range(1, 101)),
        'case_min_max': (min(cases), max(cases)) if cases else None,
        'md_n': len(mds),
        'md_exts': exts,
        'has_runner': 'model_dir/runner.py' in names,
        'has_weights': 'model_dir/your_model.pt' in names,
        'md_names': [n.replace('model_dir/', '') for n in mds],
    }


for label, group in (('【现在待交】D:/models/upload/', NOW),
                     ('【老世界成功过】提交历史/', OLD)):
    print('=' * 96)
    print(label)
    print('=' * 96)
    for p in group:
        i = info(p)
        print(f'\n{p.name}   ({p.stat().st_size/2**30:.2f} GiB)')
        print(f'  顶层: {i["top"]}')
        print(f'  output_dir: {i["out_n"]} 项 | case1..100 齐全={i["case_ok"]} {i["case_min_max"]}')
        print(f'  model_dir : {i["md_n"]} 项 | runner.py={i["has_runner"]} your_model.pt={i["has_weights"]}')
        print(f'  model_dir 扩展名分布: {i["md_exts"]}')
        if i['md_n'] <= 8:
            print(f'  model_dir 全部内容: {i["md_names"]}')
        else:
            print(f'  model_dir 前 6 项: {i["md_names"][:6]} ...')
