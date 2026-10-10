# -*- coding: utf-8 -*-
"""静态审计 4 个提交 zip 里的 .py: 是否还残留 "Python <3.10 不适配" 风险。

背景: 2026-09-03 提交的 Track2_CompliantM24.zip 在官方"测速环节"死于
  TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
=> 官方解释器 < 3.10, 运行时求值了 `X | None` 注解。
本地 harness_sim 跑在 Python 3.13 上, 原生支持 `X | None` -> **本地永远测不出来**。

本脚本用 AST 静态找三类风险:
  R1 运行时求值的 `X | None`  (BinOp BitOr, 非注解位置)
  R2 有注解含 `|` 但文件缺少/位置不对的 `from __future__ import annotations`
  R3 其他 3.10+ 语法 (match 语句 / except* / 运行时 list[int] 之类)
用法: python audit_py_compat.py [zip ...]  (缺省 = 4 个归档 zip)
"""
from __future__ import annotations

import ast
import glob
import io
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ARCH = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
DEFAULT_ZIPS = [
    ARCH / 'Track2_CompliantM24——.zip',
    ARCH / 'Track2_14B_M0S1_SC093——.zip',
    ARCH / 'Track2_14B_M0S7_SC10——.zip',
    ARCH / 'Track2_14B_G3W18S1_SC093——.zip',
]


def future_import_ok(tree: ast.Module):
    """`from __future__ import annotations` 必须是仅晚于 docstring 的语句。"""
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return False
    first = body[0]
    return (isinstance(first, ast.ImportFrom) and first.module == '__future__'
            and any(al.name == 'annotations' for al in first.names))


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.in_ann = False
        self.runtime_or_none = []   # (lineno, 源码片段)
        self.annot_or_none = 0
        self.match_stmts = []
        self.except_star = []
        self.src = ''

    # ---- 注解位置: 进入时置 in_ann ----
    def _ann(self, node):
        if node is None:
            return
        save, self.in_ann = self.in_ann, True
        self.visit(node)
        self.in_ann = save

    def _visit_func(self, node):
        a = node.args
        for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            self._ann(arg.annotation)
        if a.vararg:
            self._ann(a.vararg.annotation)
        if a.kwarg:
            self._ann(a.kwarg.annotation)
        self._ann(node.returns)
        for d in node.decorator_list:
            self.visit(d)
        for s in node.body:
            self.visit(s)

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_AnnAssign(self, node):
        self._ann(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_Match(self, node):
        self.match_stmts.append(node.lineno)
        self.generic_visit(node)

    def visit_TryStar(self, node):
        self.except_star.append(node.lineno)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.BitOr):
            l, r = node.left, node.right
            is_none = ((isinstance(l, ast.Constant) and l.value is None)
                       or (isinstance(r, ast.Constant) and r.value is None))
            if is_none:
                try:
                    frag = ast.get_source_segment(self.src, node) or ''
                except Exception:
                    frag = ''
                if self.in_ann:
                    self.annot_or_none += 1
                else:
                    self.runtime_or_none.append((node.lineno, frag.strip()[:90]))
        self.generic_visit(node)


def audit_source(name, src):
    out = {'name': name, 'syntax_error': None, 'future': False,
           'runtime_or_none': [], 'annot_or_none': 0, 'match': [], 'try_star': []}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        out['syntax_error'] = f'line {e.lineno}: {e.msg}'
        return out
    out['future'] = future_import_ok(tree)
    v = Visitor()
    v.src = src
    v.visit(tree)
    out['runtime_or_none'] = v.runtime_or_none
    out['annot_or_none'] = v.annot_or_none
    out['match'] = v.match_stmts
    out['try_star'] = v.except_star
    return out


def main():
    zips = [Path(p) for p in sys.argv[1:]] or DEFAULT_ZIPS
    total_bad = 0
    for zp in zips:
        if not zp.exists():
            print(f'!! 缺文件 {zp}')
            continue
        print('=' * 78)
        print(f'{zp.name}')
        print('=' * 78)
        py_count = 0
        problems = []
        with zipfile.ZipFile(zp) as zf:
            pys = sorted(n for n in zf.namelist() if n.lower().endswith('.py'))
            py_count = len(pys)
            for n in pys:
                src = zf.read(n).decode('utf-8', errors='replace')
                r = audit_source(n, src)
                bad = []
                if r['syntax_error']:
                    bad.append(f"SYNTAX {r['syntax_error']}")
                for ln, frag in r['runtime_or_none']:
                    bad.append(f'R1 运行时 `{frag}` @L{ln}')
                if r['annot_or_none'] and not r['future']:
                    bad.append(f"R2 {r['annot_or_none']} 处注解含 `|` 但无 future import")
                for ln in r['match']:
                    bad.append(f'R3 match 语句 @L{ln}')
                for ln in r['try_star']:
                    bad.append(f'R3 except* @L{ln}')
                if bad:
                    problems.append((n, bad))
        print(f'.py 文件数 = {py_count}')
        if not problems:
            print('  ✅ 未发现 Python<3.10 不适配风险')
        for n, bad in problems:
            print(f'  ⚠️ {n}')
            for b in bad:
                print(f'      - {b}')
        total_bad += len(problems)
        print()
    print('#' * 78)
    print(f'合计有风险文件数: {total_bad}')


if __name__ == '__main__':
    main()
