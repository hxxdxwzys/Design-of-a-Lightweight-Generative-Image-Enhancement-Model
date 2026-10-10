# -*- coding: utf-8 -*-
"""修复 v2: BOM 安全的重建 (2026-09-07)
对 model_dir 全部 .py: 剥离 BOM → 移除误插/已插的 future 行 → 用 tokenize 正确重插到 docstring 之后
写回: 原文件有 BOM 则用 utf-8-sig (BOM 回文件头)
"""
import io
import sys
import tokenize
from pathlib import Path

FUTURE = 'from __future__ import annotations'


def find_insert_line(lines):
    """返回应插入 future 的行号(0-based, 插在该行前); 跳过 shebang/coding/注释/空行/docstring"""
    src = ''.join(lines)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return 0
    first = None
    for t in toks:
        if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE,
                      tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            continue
        first = t
        break
    if first is None:
        return 0
    if first.type == tokenize.STRING:
        s = first.string
        # 模块级 docstring (独立字符串语句) → future 插在它之后
        return first.end[0]  # end row (1-based) → 插在第 end 行前 (0-based end)
    return first.start[0] - 1  # 插在首语句行前


def fix_file(path: Path):
    raw = path.read_bytes()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    text = raw.decode('utf-8-sig') if has_bom else raw.decode('utf-8', errors='replace')
    lines = text.splitlines(keepends=True)
    # 移除已有 future 行 (可能带 \r\n)
    lines = [l for l in lines if l.strip().rstrip('\r\n') != FUTURE]
    # 计算插入行
    ins = find_insert_line(lines)
    lines.insert(ins, FUTURE + '\n')
    out = ''.join(lines)
    # 写回 (保留原换行风格近似: 若原文件 \r\n 则未来行用 \r\n)
    if '\r\n' in text:
        out = out.replace('\n', '\r\n')
    path.write_bytes((('\xef\xbb\xbf' if has_bom else '') + out).encode('utf-8'))
    return True


def main():
    root = Path(sys.argv[1])
    n = 0
    for p in sorted(root.rglob('*.py')):
        fix_file(p)
        n += 1
    print(f'rebuilt {n} files (BOM-safe)')


if __name__ == '__main__':
    main()
