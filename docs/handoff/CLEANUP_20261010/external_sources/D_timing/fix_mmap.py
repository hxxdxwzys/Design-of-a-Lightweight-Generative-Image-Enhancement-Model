# -*- coding: utf-8 -*-
"""M24 runner 的 mmap 补丁 + 5 个 runner 加固状态核验 (2026-09-11)"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TARGETS = {
    'M24': Path(r'D:/models/final_compliant/model_dir/runner.py'),
    'm0s1': Path(r'D:/models/final_compliant_14b/model_dir/runner.py'),
    'm0s7_sc10': Path(r'D:/models/final_compliant_m0s7/model_dir/runner.py'),
    'g3w18s1': Path(r'D:/models/final_compliant_g3w18s1/model_dir/runner.py'),
    'm0s7_s097': Path(r'D:/models/wave2_m0s7_s097/pack_model_dir/runner.py'),
}

OLD = "map_location='cpu', weights_only=False)"
NEW = "map_location='cpu', weights_only=False, mmap=True)"

for tag, p in TARGETS.items():
    if not p.exists():
        print(f'{tag:<10} !! 缺文件 {p}'); continue
    t = p.read_text(encoding='utf-8')
    if 'mmap=True' not in t:
        t = t.replace(OLD, NEW, 1)
        p.write_text(t, encoding='utf-8', newline='')
        try:
            py_compile.compile(str(p), doraise=True)
            note = '已补 mmap + 编译OK'
        except Exception as e:
            note = f'!! 编译失败 {e}'
    else:
        note = '已含 mmap'
    t = p.read_text(encoding='utf-8')
    print(f'{tag:<10} dynamo={"✅" if "TORCHDYNAMO_DISABLE" in t else "❌"} '
          f'mmap={"✅" if "mmap=True" in t else "❌"} '
          f'回退={"✅" if "mmap 不可用" in t else "❌"} | {note}')
