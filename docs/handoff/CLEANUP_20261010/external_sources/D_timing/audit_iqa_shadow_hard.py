# -*- coding: utf-8 -*-
"""更严格的 IQA 遮蔽审计 (2026-09-11): 模拟"官方 harness 在我们 runner 之后再插一次 model_dir"

背景: 现有 audit_iqa_shadow.py 只模拟"先插 model_dir 再 import runner"（我们的补丁能处理）。
但官方 harness 完全可能在【加载完 runner 之后】、加载 IQA 评测器之前再 `sys.path.insert(0, model_dir)`
——此时我们 runner 的清理已失效, 如果 model_dir 里还躺着 timm/einops 桩包, 依旧会遮蔽。

本审计 = audit_iqa_shadow + 第 ③ 步"再插一次", 即对抗性最强的情形。
用法: python audit_iqa_shadow_hard.py <model_dir> [more...]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROBE = r'''
import os, sys
MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, MD)          # ① 官方 harness: 插 model_dir 以便 import runner
import runner                    # ② 我们的 runner (内部会做清理)
sys.path.insert(0, MD)          # ③ 关键: harness 之后再插一次 (对抗情形)
import timm, einops
def outside(m):
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    return not f.startswith(MD + os.sep)
try:
    from timm.models.vision_transformer import Block   # noqa: F401
    block_ok = True
except Exception as e:
    block_ok = False
ok = outside(timm) and outside(einops) and block_ok
print("HARD pass=%s timm_outside=%s einops_outside=%s block_ok=%s" % (ok, outside(timm), outside(einops), block_ok))
print("HARD timm=", os.path.abspath(getattr(timm, "__file__", "")))
print("HARD einops=", os.path.abspath(getattr(einops, "__file__", "")))
sys.exit(0 if ok else 1)
'''


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [
        Path(r'D:/models/final_compliant/model_dir'),
        Path(r'D:/models/final_compliant_14b/model_dir'),
        Path(r'D:/models/final_compliant_g3w18s1/model_dir'),
        Path(r'D:/models/final_compliant_m0s7/model_dir'),
        Path(r'D:/models/wave2_m0s7_s097/pack_model_dir'),
    ]
    probe = Path(r'D:/tmp_timing/_probe_iqa_hard.py')
    probe.write_text(PROBE, encoding='utf-8')
    allok = True
    for md in dirs:
        if not (md / 'runner.py').exists():
            print(f'-- 跳过(无 runner.py): {md}')
            continue
        has_stub = (md / 'timm').is_dir() or (md / 'einops').is_dir()
        r = subprocess.run([sys.executable, str(probe), str(md)],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=1800, cwd=r'C:/models/finals')
        out = (r.stdout or '') + (r.stderr or '')
        hit = [ln for ln in out.splitlines() if ln.startswith('HARD')]
        print(f'===== {md}')
        print(f'  含桩包(timm/einops 在 model_dir 根): {has_stub}')
        for ln in hit:
            print('  ' + ln)
        verdict = 'PASS' if r.returncode == 0 else 'FAIL'
        print(f'  ==> HARD_AUDIT_{verdict}')
        allok = allok and r.returncode == 0
        print()
    print('HARD_ALL_PASS' if allok else 'HARD_ANY_FAIL')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
