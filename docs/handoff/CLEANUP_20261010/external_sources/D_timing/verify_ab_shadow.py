# -*- coding: utf-8 -*-
"""决定性 A/B 核验: 旧 runner(未修) vs 新 runner(已修) 的 IQA 遮蔽审计 (2026-09-11)
做法: 用 final_compliant 的 .py 结构造两个临时 model_dir, 一个放旧 runner (从提交历史 zip 提取),
      一个放新 runner; 分别跑 audit_iqa_shadow.py。期望: 旧 FAIL / 新 PASS。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SRC = Path(r'D:/models/final_compliant/model_dir')
NEW_RUNNER = SRC / 'runner.py'
OLD_ZIP = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史/Track2_CompliantM24——.zip')
TMP = Path(r'D:/tmp_timing/_audit_ab')


def build(tag, runner_bytes):
    d = TMP / tag / 'model_dir'
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in SRC.rglob('*.py'):
        rel = f.relative_to(SRC)
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / rel)
    (d / 'runner.py').write_bytes(runner_bytes)
    return d


def main():
    if not OLD_ZIP.exists():
        print(f'!! 找不到旧包 {OLD_ZIP}')
        return 1
    with zipfile.ZipFile(OLD_ZIP) as z:
        old_runner = z.read('model_dir/runner.py')
    print(f'旧 runner: {len(old_runner)} bytes (来自 {OLD_ZIP.name})')
    print(f'新 runner: {NEW_RUNNER.stat().st_size} bytes (已打补丁)')
    print()

    res = {}
    for tag, data in (('old_unpatched', old_runner), ('new_patched', NEW_RUNNER.read_bytes())):
        d = build(tag, data)
        r = subprocess.run([sys.executable, r'D:/tmp_timing/audit_iqa_shadow.py', str(d)],
                           capture_output=True, text=True, timeout=1800)
        out = (r.stdout or '') + (r.stderr or '')
        print(f'===== {tag} =====')
        print(out.strip()[-900:])
        print()
        res[tag] = {'rc': r.returncode,
                    'pass': 'IQA_SHADOW_AUDIT_PASS' in out,
                    'fail': 'IQA_SHADOW_AUDIT_FAIL' in out}
    print('=== A/B 结论 ===')
    o, n = res['old_unpatched'], res['new_patched']
    verdict = ('A/B 核验通过: 旧包被拦下(FAIL) + 新包通过(PASS) -> 审计有鉴别力, 修复有效'
               if (o['fail'] and n['pass']) else 'A/B 核验异常: ' + json.dumps(res))
    print('  ' + verdict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
