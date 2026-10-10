# -*- coding: utf-8 -*-
"""结构性加固: 把桩包挪进 _vendor/ , 让"model_dir 被插到 sys.path 最前"也遮蔽不了官方依赖

问题: model_dir 根目录下的 timm/einops/diffusers/fairscale 是裁剪桩。只要 model_dir 出现在
sys.path 最前(官方 harness 任意时刻插入), pyiqa 的 MANIQA 就会命中桩包 ->
"Failed to load IQAEvaluator"。现有补丁只在【我们 runner 运行期间】治理, 之后无法阻止再次插入。

加固: 桩包移到 model_dir/_vendor/ ; runner 导入我们自己的模块时把 _vendor 置顶(行为与现在一致),
      导入完成后撤掉 model_dir 与 _vendor, 再把 _vendor 追加到末尾作回落(真包缺失时才用桩)。
      => model_dir 根目录不再含 timm/einops, 无论谁把 model_dir 插到最前都不会遮蔽真包。

用法: python harden_vendor.py [--apply] <model_dir> [more...]
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

APPLY = '--apply' in sys.argv
STUBS = ('timm', 'einops', 'diffusers', 'fairscale')

OLD_HEAD = re.compile(
    r'_MODEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n'
    r'_PATH_BEFORE = list\(sys\.path\)\n'
    r'if _MODEL_DIR not in sys\.path:\n'
    r'    sys\.path\.insert\(0, _MODEL_DIR\)\n')

NEW_HEAD = (
    "_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))\n"
    "_VENDOR_DIR = os.path.join(_MODEL_DIR, '_vendor')\n"
    "_PATH_BEFORE = list(sys.path)\n"
    "# 桩包集中在 _vendor/ (2026-09-11 加固): 导入本模块期间置顶使用, 之后撤掉;\n"
    "# 这样 model_dir 根目录不含 timm/einops, 官方 harness 再把 model_dir 插到最前也不会遮蔽真包。\n"
    "if os.path.isdir(_VENDOR_DIR):\n"
    "    sys.path.insert(0, _VENDOR_DIR)\n"
    "if _MODEL_DIR not in sys.path:\n"
    "    sys.path.insert(0, _MODEL_DIR)\n")

OLD_TAIL = re.compile(
    r'_MD_ABS = os\.path\.abspath\(_MODEL_DIR\)\n'
    r'(?:#.*\n)*'
    r'sys\.path\[:\] = \[p for p in _PATH_BEFORE if os\.path\.abspath\(p\) != _MD_ABS\]\n'
    r'for _n in \[n for n in sys\.modules if n\.split\(\'\.\'\)\[0\] in\n'
    r"           \('timm', 'einops', 'diffusers', 'fairscale'\)\]:\n"
    r"    _f = getattr\(sys\.modules\[_n\], '__file__', None\) or ''\n"
    r'    if _f and os\.path\.abspath\(_f\)\.startswith\(_MD_ABS \+ os\.sep\):\n'
    r'        del sys\.modules\[_n\]\n'
    r'if _MODEL_DIR not in sys\.path:\n'
    r'    sys\.path\.append\(_MODEL_DIR\)\n')

NEW_TAIL = (
    "_MD_ABS = os.path.abspath(_MODEL_DIR)\n"
    "_VD_ABS = os.path.abspath(_VENDOR_DIR)\n"
    "sys.path[:] = [p for p in _PATH_BEFORE\n"
    "               if os.path.abspath(p) not in (_MD_ABS, _VD_ABS)]\n"
    "for _n in [n for n in sys.modules if n.split('.')[0] in\n"
    "           ('timm', 'einops', 'diffusers', 'fairscale')]:\n"
    "    _f = getattr(sys.modules[_n], '__file__', None) or ''\n"
    "    _fa = os.path.abspath(_f) if _f else ''\n"
    "    if _fa.startswith(_MD_ABS + os.sep) or _fa.startswith(_VD_ABS + os.sep):\n"
    "        del sys.modules[_n]\n"
    "if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:\n"
    "    sys.path.append(_VENDOR_DIR)   # 回落: 真包缺失时才用我们的桩\n"
    "if _MODEL_DIR not in sys.path:\n"
    "    sys.path.append(_MODEL_DIR)\n")


def harden(md: Path):
    log = []
    # 1) 移动桩包
    vendor = md / '_vendor'
    moved = []
    for name in STUBS:
        src = md / name
        if src.is_dir():
            vendor.mkdir(exist_ok=True)
            dst = vendor / name
            if dst.exists():
                shutil.rmtree(dst)
            if APPLY:
                shutil.move(str(src), str(dst))
            moved.append(name)
    log.append(f'桩包移入 _vendor/: {moved if moved else "(无, 可能已加固)"}')

    # 2) 补丁 runner.py
    rp = md / 'runner.py'
    text = rp.read_text(encoding='utf-8')
    if '_VENDOR_DIR' in text:
        log.append('runner.py: 已加固')
        return log
    if not OLD_HEAD.search(text):
        log.append('runner.py: !! 未找到旧头部锚点, 跳过')
        return log
    if not OLD_TAIL.search(text):
        log.append('runner.py: !! 未找到旧尾部锚点, 跳过')
        return log
    text = OLD_HEAD.sub(NEW_HEAD, text, count=1)
    text = OLD_TAIL.sub(NEW_TAIL, text, count=1)
    if APPLY:
        rp.write_text(text, encoding='utf-8', newline='')
    log.append('runner.py: 头部+尾部已替换为 _vendor 版')
    return log


def main():
    targets = [Path(a) for a in sys.argv[1:] if not a.startswith('--')]
    if not targets:
        print('用法: harden_vendor.py [--apply] <model_dir> [...]')
        return 2
    for md in targets:
        print(f'===== {md}')
        if not (md / 'runner.py').exists():
            print('  !! 无 runner.py, 跳过'); continue
        for line in harden(md):
            print('  ' + line)
    print('\n(干跑; 加 --apply 生效)' if not APPLY else '\n(已写入)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
