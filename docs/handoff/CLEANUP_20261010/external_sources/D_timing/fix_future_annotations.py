# -*- coding: utf-8 -*-
"""批量修复: model_dir 所有 .py 加 from __future__ import annotations (兼容官方 <Py3.10)
规则: 插入到 docstring 之后、第一个语句之前 (tokenize 定位, 幂等)
用法: python fix_future_annotations.py <dir>
"""
import io
import sys
import tokenize
from pathlib import Path

FUTURE = 'from __future__ import annotations\n'


def needs_fix(src: str) -> bool:
    return 'from __future__ import annotations' not in src


def insert_pos(src: str) -> int:
    """返回插入位置 (docstring 结束后的行首偏移); 无 docstring 则在首个语句前"""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return 0
    # 找第一个有意义 token
    first_stmt = None
    for t in toks:
        if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE,
                      tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            continue
        first_stmt = t
        break
    if first_stmt is None:
        return 0
    # 若首语句是 docstring (模块级 STRING), 跳过其结束
    if first_stmt.type == tokenize.STRING:
        s = first_stmt.string
        if s.startswith(('"""', "'''")) and (s.count('"""') >= 2 or s.count("'''") >= 2
                                             or '\n' in s):
            # 单 token 含整个 docstring? 找其 end 行
            end_row = first_stmt.end[0]
            # 定位 end_row 行首
            lines = src.splitlines(keepends=True)
            pos = sum(len(x) for x in lines[:end_row])
            return pos
        return first_stmt.start[1] if first_stmt.start[0] == 1 else 0
    # 首个语句行首
    lines = src.splitlines(keepends=True)
    pos = sum(len(x) for x in lines[:first_stmt.start[0] - 1])
    # 若该行有前导缩进则插行首; 模块级应无缩进
    return pos


def fix_file(path: Path):
    src = path.read_text(encoding='utf-8', errors='replace')
    if not needs_fix(src):
        return False
    pos = insert_pos(src)
    new_src = src[:pos] + FUTURE + src[pos:]
    path.write_text(new_src, encoding='utf-8')
    return True


def main():
    root = Path(sys.argv[1])
    fixed = []
    for p in sorted(root.rglob('*.py')):
        if fix_file(p):
            fixed.append(str(p))
    print(f'fixed {len(fixed)} files:')
    for f in fixed:
        print(' ', f)


if __name__ == '__main__':
    main()
