# -*- coding: utf-8 -*-
"""给 runner.py 打补丁: 阻止自带裁剪桩遮蔽官方环境的真包 (2026-09-11)

问题: model_dir 自带 timm(仅3文件,无 Block)/einops(仅3种pattern) 等裁剪桩,
runner.py 用 sys.path.insert(0, _MODEL_DIR) 置顶 -> 官方同进程加载 IQA 评测器
(pyiqa: CLIPIQA/MUSIQ/MANIQA/NIQE) 时命中桩包, 报
  "Failed to load IQAEvaluator" / ImportError: cannot import name 'Block' from 'timm...'

修法: 置顶加载我们自己的包 -> 立刻撤掉置顶 + 清掉 sys.modules 里来自本目录的
timm/einops/diffusers/fairscale -> 真包生效; 本目录仅追加末尾作回落 (都没有时才用桩)。

用法: D:/Python/python.exe patch_runner_shadow.py [--apply]
不加 --apply 只做 dry-run 报告。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TARGETS = [
    Path(r'D:/models/final_compliant/model_dir/runner.py'),
    Path(r'D:/models/final_compliant_14b/model_dir/runner.py'),
    Path(r'D:/models/final_compliant_g3w18s1/model_dir/runner.py'),
    Path(r'D:/models/final_compliant_m0s7/model_dir/runner.py'),
]

ANCHOR_OLD = re.compile(
    r'_MODEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\s*\n'
    r'if _MODEL_DIR not in sys\.path:\s*\n'
    r'    sys\.path\.insert\(0, _MODEL_DIR\)\s*\n')

ANCHOR_NEW = (
    "_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))\n"
    "_PATH_BEFORE = list(sys.path)\n"
    "if _MODEL_DIR not in sys.path:\n"
    "    sys.path.insert(0, _MODEL_DIR)\n")

IMPORT_RE = re.compile(r'^from (?:models|dinov2|vae_shim)[\w.]* import .*# noqa: E402[ \t]*$', re.M)

RESTORE = '''

# ── 交还 import 环境 (关键: 官方 IQA 评测器与我们在同一进程) ────────────────
# 本目录自带的是【裁剪桩】: timm 仅 3 个文件 (无 Block); einops 仅支持 3 种 pattern。
# 官方质量评测用 pyiqa (CLIPIQA/MUSIQ/MANIQA/NIQE), 它会 import timm/einops; 若命中桩包
# 就报 "Failed to load IQAEvaluator" (2026-09-11 实测 MANIQA:
# ImportError: cannot import name 'Block' from 'timm.models.vision_transformer')。
# 处置: 撤掉置顶路径, 并清掉 sys.modules 中来自本目录的同名包, 让官方环境里的真包生效;
#       本目录仅追加到 sys.path 末尾作为回落 (真包缺失时才用我们的桩)。
sys.path[:] = _PATH_BEFORE
_MD_ABS = os.path.abspath(_MODEL_DIR)
for _n in [n for n in sys.modules if n.split('.')[0] in
           ('timm', 'einops', 'diffusers', 'fairscale')]:
    _f = getattr(sys.modules[_n], '__file__', None) or ''
    if _f and os.path.abspath(_f).startswith(_MD_ABS + os.sep):
        del sys.modules[_n]
if _MODEL_DIR not in sys.path:
    sys.path.append(_MODEL_DIR)
'''


def patch(text):
    if '_PATH_BEFORE' in text:
        return text, 'already patched'
    newline = '\r\n' if '\r\n' in text else '\n'
    norm = text.replace('\r\n', '\n').replace('\r', '\n')
    if not ANCHOR_OLD.search(norm):
        return text, 'ANCHOR NOT FOUND'
    if not IMPORT_RE.search(norm):
        return text, 'IMPORT BLOCK NOT FOUND'
    norm = ANCHOR_OLD.sub(ANCHOR_NEW, norm, count=1)
    matches = list(IMPORT_RE.finditer(norm))
    last = matches[-1]
    norm = norm[:last.end()] + RESTORE + norm[last.end():]
    if newline == '\r\n':
        norm = norm.replace('\n', '\r\n')
    return norm, f'patched ({len(matches)} import lines)'


def main():
    apply = '--apply' in sys.argv
    for p in TARGETS:
        if not p.exists():
            print(f'!! 缺文件 {p}')
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        new, status = patch(text)
        print(f'{p.parent.parent.name:26s} {status}')
        if apply and new != text:
            p.write_text(new, encoding='utf-8', newline='')
            print('    -> 已写入')
    if not apply:
        print('\n(dry-run; 加 --apply 才写入)')


if __name__ == '__main__':
    main()
