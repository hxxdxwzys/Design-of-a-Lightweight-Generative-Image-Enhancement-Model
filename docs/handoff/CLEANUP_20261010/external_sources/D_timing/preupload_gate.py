# -*- coding: utf-8 -*-
"""上传前总门禁 (2026-09-11): 一次性堵住我们已经踩过的三类提交问题

历史踩坑与对应检查:
  ① 文件名含中文 -> 上传失败 (曾浪费 3 次机会)
       -> 检查 zip 文件名与内部所有路径名纯 ASCII
  ② TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
     (官方解释器 Py<3.10 运行时求值 X | None 注解; 09-03 实际踩到)
       -> audit_py_compat (AST 静态审计, 本地 python 是 3.13 测不出)
  ③ Evaluation failed during quality: Failed to load IQAEvaluator
     (model_dir 自带裁剪桩 timm/einops 遮蔽官方 pyiqa 依赖; 09-11 实际踩到)
       -> audit_iqa_shadow (模拟官方先插 model_dir 再 import runner, 查真包是否被遮蔽)

外加基础检查: zip 完整性 / 结构(100 jpg + runner.py + your_model.pt) / 体积<=10GB /
             接口可加载并 512 推理 (harness_sim_zip) / 出图尺寸与可解码性。

用法:
  python preupload_gate.py <zip> [more.zip ...]
  python preupload_gate.py --stage-upload <zip> --upload-name <纯ASCII名>   # 归位到 D:/models/upload/
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
UPLOAD_DIR = Path(r'D:/models/upload')
SIM_DIR = Path(r'D:/models/_harness_zip_sim')
MAX_GB = 10.0
ASCII_RE = re.compile(r'^[\x20-\x7e]+$')


def check_ascii_name(p: Path):
    bad = [c for c in p.name if ord(c) > 126]
    if bad:
        return False, f'文件名含非 ASCII 字符: {"".join(bad)!r} -> 会重演「文件名含中文」失败'
    return True, f'文件名纯 ASCII: {p.name}'


def check_inner_names(p: Path):
    with zipfile.ZipFile(p) as z:
        bad = [n for n in z.namelist() if not ASCII_RE.match(n)]
    if bad:
        return False, f'zip 内有 {len(bad)} 个非 ASCII 路径 (前3: {bad[:3]})'
    return True, 'zip 内路径全部 ASCII'


def check_structure(p: Path):
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        bad = z.testzip()
    if bad is not None:
        return False, f'zip 损坏 @ {bad}'
    jpgs = [n for n in names if n.startswith('output_dir/') and n.lower().endswith('.jpg')]
    mds = [n for n in names if n.startswith('model_dir/')]
    problems = []
    if len(jpgs) != 100:
        problems.append(f'output_dir 有 {len(jpgs)} 张 jpg (需 100)')
    for must in ('model_dir/runner.py', 'model_dir/your_model.pt'):
        if must not in names:
            problems.append(f'缺 {must}')
    if problems:
        return False, '; '.join(problems)
    return True, f'testzip OK; output_dir {len(jpgs)} jpg; model_dir {len(mds)} 项'


def check_size(p: Path):
    gb = p.stat().st_size / 2 ** 30
    if gb > MAX_GB:
        return False, f'{gb:.2f} GiB > {MAX_GB} GB 上限'
    return True, f'{gb:.2f} GiB (上限 {MAX_GB} GB)'


def check_jpg_dims(p: Path, n=3):
    """抽 n 张出图检查尺寸(应为 4K)与可解码性"""
    try:
        from PIL import Image
    except Exception as e:
        return True, f'SKIP (PIL 不可用: {e})'
    with zipfile.ZipFile(p) as z:
        jpgs = sorted(n_ for n_ in z.namelist()
                      if n_.startswith('output_dir/') and n_.lower().endswith('.jpg'))
        if not jpgs:
            return False, '无 jpg 可查'
        picks = [jpgs[0], jpgs[len(jpgs) // 2], jpgs[-1]][:n]
        info = []
        for nm in picks:
            import io
            with Image.open(io.BytesIO(z.read(nm))) as im:
                w, h = im.size
                info.append(f'{Path(nm).name} {w}x{h}')
                if min(w, h) < 2000:
                    return False, f'{Path(nm).name} 尺寸 {w}x{h} 不是 4K 量级'
        return True, '; '.join(info)


def run_tool(script, args, cwd=None):
    # 2026-09-11 修: 未指定 encoding 时 Windows 用 locale(GBK) 解码子进程 UTF-8 输出 ->
    # UnicodeDecodeError 被吞 -> 审计输出为空 -> ② 语法兼容 假失败。必须显式 utf-8。
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([PY, str(TOOLS / script), *map(str, args)],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=3600, cwd=cwd, env=env)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zips', nargs='+')
    ap.add_argument('--stage-upload', action='store_true',
                    help='先把 zip 归位到 D:/models/upload/<upload-name> 再对归位后的文件做门禁')
    ap.add_argument('--upload-name', default=None, help='纯 ASCII 上传名 (含 .zip)')
    a = ap.parse_args()

    ok_all = True
    for zp in a.zips:
        src = Path(zp)
        print('=' * 74)
        print(f'门禁: {src}')
        print('=' * 74)
        if not src.exists():
            print('  ❌ 文件不存在'); ok_all = False; continue

        target = src
        if a.stage_upload:
            if not a.upload_name:
                print('  ❌ --stage-upload 需同时给 --upload-name'); ok_all = False; continue
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            target = UPLOAD_DIR / a.upload_name
            if target.exists():
                target.unlink()
            try:
                os.link(src, target)          # 硬链接: 零额外磁盘
                how = '硬链接'
            except OSError:
                shutil.copy2(src, target)
                how = '复制'
            print(f'  [归位] {how} -> {target}')

        results = []
        for name, fn in (
            ('① 文件名 ASCII', lambda: check_ascii_name(target)),
            ('   内部路径 ASCII', lambda: check_inner_names(target)),
            ('   结构与完整性', lambda: check_structure(target)),
            ('   体积', lambda: check_size(target)),
            ('   出图尺寸', lambda: check_jpg_dims(target)),
        ):
            try:
                ok, msg = fn()
            except Exception as e:
                ok, msg = False, f'{type(e).__name__}: {e}'
            results.append((name, ok, msg))
            print(f'  {"✅" if ok else "❌"} {name}: {msg}')

        # ② 官方 Py<3.10 的 |None
        rc, out = run_tool('audit_py_compat.py', [target])
        ok2 = '合计有风险文件数: 0' in out
        results.append(('② 语法兼容 (|None)', ok2, 'AST 审计 0 处风险' if ok2 else '发现风险'))
        print(f'  {"✅" if ok2 else "❌"} ② 语法兼容: {"0 处风险" if ok2 else "有风险!"}')
        if not ok2:
            print('\n'.join('     ' + ln for ln in out.strip().splitlines()[-12:]))

        # 接口 (harness sim: 解压 + import + 512 infer)
        rc, out = run_tool('harness_sim_zip.py', [target])
        ok_h = 'infer OK' in out
        results.append(('   接口/512 推理', ok_h, 'infer OK' if ok_h else '失败'))
        print(f'  {"✅" if ok_h else "❌"}   接口/512 推理: {"infer OK" if ok_h else "失败"}')
        if not ok_h:
            print('\n'.join('     ' + ln for ln in out.strip().splitlines()[-10:]))

        # ③ IQA 依赖遮蔽 (用 harness sim 解压出来的 model_dir)
        if SIM_DIR.exists():
            rc, out = run_tool('audit_iqa_shadow.py', [SIM_DIR / 'model_dir'])
            ok3 = 'IQA_SHADOW_AUDIT_PASS' in out
            results.append(('③ IQA 遮蔽', ok3, 'PASS' if ok3 else 'FAIL'))
            print(f'  {"✅" if ok3 else "❌"} ③ IQA 遮蔽: {"PASS" if ok3 else "FAIL"}')
            if not ok3:
                print('\n'.join('     ' + ln for ln in out.strip().splitlines()[-8:]))
        else:
            results.append(('③ IQA 遮蔽', False, 'SKIP(无解压目录)'))
            print('  ❌ ③ IQA 遮蔽: SKIP — harness sim 未产出解压目录')

        # ④ IQA 隐蔽【对抗版】(2026-09-11 新增): 模拟官方 harness 在 import 完 runner 之后
        #    再把 model_dir 插到 sys.path 最前 -> 只有把桩包挪进 _vendor/ 的结构才能过
        if SIM_DIR.exists():
            rc, out = run_tool('audit_iqa_shadow_hard.py', [SIM_DIR / 'model_dir'])
            ok4 = 'HARD_AUDIT_PASS' in out or 'HARD_ALL_PASS' in out
            results.append(('④ IQA 遮蔽(对抗)', ok4, 'PASS' if ok4 else 'FAIL'))
            print(f'  {"✅" if ok4 else "❌"} ④ IQA 遮蔽(对抗): {"PASS" if ok4 else "FAIL"}')
            if not ok4:
                print('     (model_dir 根目录仍含 timm/einops 桩包 -> 官方若再插一次 model_dir 会重演失败)')
                print('\n'.join('     ' + ln for ln in out.strip().splitlines()[-6:]))
        else:
            results.append(('④ IQA 遮蔽(对抗)', False, 'SKIP(无解压目录)'))
            print('  ❌ ④ IQA 遮蔽(对抗): SKIP — harness sim 未产出解压目录')

        # ⑤ 官方格式严格审计 (2026-09-11 新增): 顶层目录 / 出图名与官方测试集严格一致 /
        #    model_dir 必备文件 / 无杂质 / 接口签名 / [-1,1] 512 实跑
        rc, out = run_tool('audit_official_format.py', [target])
        ok5 = 'FORMAT_PASS' in out and 'FORMAT_FAIL' not in out
        results.append(('⑤ 官方格式严格审计', ok5, 'PASS' if ok5 else 'FAIL'))
        print(f'  {"✅" if ok5 else "❌"} ⑤ 官方格式严格审计: {"PASS" if ok5 else "FAIL"}')
        if not ok5:
            print('\n'.join('     ' + ln for ln in out.strip().splitlines()[-10:]))

        bad = [n for n, ok, _ in results if not ok]
        verdict = 'PREUPLOAD_GATE_PASS' if not bad else f'PREUPLOAD_GATE_FAIL ({", ".join(bad)})'
        print(f'  ==> {verdict}')
        ok_all = ok_all and not bad

    print('=' * 74)
    print('总体: ' + ('✅ 全部通过，可以上传' if ok_all else '❌ 存在问题，不要上传'))
    print('PREUPLOAD_ALL_PASS' if ok_all else 'PREUPLOAD_ANY_FAIL')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
