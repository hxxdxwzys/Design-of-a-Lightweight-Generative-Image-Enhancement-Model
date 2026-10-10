# -*- coding: utf-8 -*-
"""MANIQA 本地可用性诊断 + 下载清单 (2026-09-11)"""
from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pyiqa

print('=== pyiqa 版本:', getattr(pyiqa, '__version__', '?'))
try:
    from pyiqa.default_model_configs import DEFAULT_CONFIGS
    cfg = DEFAULT_CONFIGS.get('maniqa', {})
    print('=== MANIQA 默认配置 ===')
    for k, v in cfg.items():
        print(f'  {k}: {str(v)[:220]}')
except Exception as e:
    print('config 读取失败:', e)

p = os.path.join(os.path.dirname(pyiqa.__file__), 'archs', 'maniqa_arch.py')
if os.path.exists(p):
    s = open(p, encoding='utf-8', errors='replace').read()
    urls = sorted(set(re.findall(r'https?://[^\s"\'()]+', s)))
    print('\n=== maniqa_arch.py 内 URL ===')
    for u in urls[:10]:
        print('  ', u)
    m = re.findall(r'(vit_\w+|timm\.create_model\([^)]*\))', s)
    print('\n=== timm backbone 线索 ===')
    for x in sorted(set(m))[:10]:
        print('  ', x)

# 已有缓存?
cache = os.path.expanduser('~/.cache/torch/hub/pyiqa')
print('\n=== 本地 pyiqa 缓存 ===')
if os.path.isdir(cache):
    for f in sorted(os.listdir(cache)):
        print('  ', f, os.path.getsize(os.path.join(cache, f)) // 1024, 'KB')
else:
    print('  (无)')
