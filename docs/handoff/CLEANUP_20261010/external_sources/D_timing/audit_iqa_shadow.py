# -*- coding: utf-8 -*-
"""审计: model_dir 里的自带包是否会遮蔽官方 IQA 评测器的依赖 (2026-09-11)

背景: 官方返回 "Evaluation failed during quality: Failed to load IQAEvaluator"。
根因候选之一(且已本地复现): model_dir 自带裁剪桩 timm(仅3文件,无 Block)/einops(仅3种
pattern), 官方同进程加载 pyiqa 的 MANIQA 时 `from timm.models.vision_transformer import Block`
命中桩包 -> ImportError -> 评测器加载失败。

本脚本模拟官方 harness 的调用方式 (先把 model_dir 插到 sys.path 最前, 再 import runner),
然后在【同一进程】里检查官方 IQA 依赖能否正常解析。

用法: python audit_iqa_shadow.py <model_dir>
退出码: 0 = PASS(不遮蔽), 1 = FAIL(遮蔽)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROBE = r'''
import os, sys
MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, MD)          # 官方 harness 的标准做法
import runner                    # noqa: F401
import timm, einops
def outside(m):
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    return not f.startswith(MD + os.sep)
ok = outside(timm) and outside(einops)
try:
    from timm.models.vision_transformer import Block   # noqa: F401
    block_ok = True
except Exception:
    block_ok = False
print("PROBE pass=%s timm_outside=%s einops_outside=%s block_ok=%s" % (ok and block_ok, outside(timm), outside(einops), block_ok))
print("PROBE timm=", os.path.abspath(getattr(timm, "__file__", "")))
print("PROBE einops=", os.path.abspath(getattr(einops, "__file__", "")))
sys.exit(0 if (ok and block_ok) else 1)
'''


def main():
    if len(sys.argv) < 2:
        print('用法: audit_iqa_shadow.py <model_dir>')
        return 2
    md = Path(sys.argv[1])
    if not (md / 'runner.py').exists():
        print(f'FAIL: {md} 下没有 runner.py')
        return 1
    probe = Path(r'D:/tmp_timing/_probe_iqa_shadow.py')
    probe.write_text(PROBE, encoding='utf-8')
    r = subprocess.run([sys.executable, str(probe), str(md)],
                       capture_output=True, text=True, timeout=900, cwd=r'C:/models/finals')
    out = (r.stdout or '') + (r.stderr or '')
    for line in out.splitlines():
        if line.startswith('PROBE'):
            print('  ' + line)
    if r.returncode == 0:
        print('IQA_SHADOW_AUDIT_PASS')
        return 0
    print('IQA_SHADOW_AUDIT_FAIL  (model_dir 的桩包会遮蔽官方 IQA 依赖 -> 官方会报 '
          '"Failed to load IQAEvaluator")')
    return 1


if __name__ == '__main__':
    sys.exit(main())
