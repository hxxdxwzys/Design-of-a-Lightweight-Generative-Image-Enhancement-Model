# -*- coding: utf-8 -*-
"""重建 5 个"结构加固版"提交包 (2026-09-11)
步骤: ① harden_vendor (桩包 -> _vendor/) ② package_generic 重打为 _v3
输出: D:/models/Track2_*_v3.zip  (upload/ 里的 _v2 暂不动, 等门禁全过后再换)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, r'D:/tmp_timing')
import harden_vendor  # noqa: E402

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')

# (model_dir 源, output_dir 源, 新 zip 名)
JOBS = [
    ('final_compliant', 'Track2_M24_v3'),
    ('final_compliant_14b', 'Track2_M0S1_SC093_v3'),
    ('final_compliant_m0s7', 'Track2_M0S7_SC10_v3'),
    ('final_compliant_g3w18s1', 'Track2_G3W18S1_SC093_v3'),
    ('wave2_m0s7_s097/pack_model_dir', 'Track2_M0S7_SC097_v3'),
]


def main():
    harden_vendor.APPLY = True
    for d, name in JOBS:
        base = MODELS / d
        # 2026-09-11 修 bug: 原写 base.parent/'output_dir' (错, 指向 D:/models/output_dir)
        md, od = base / 'model_dir', base / 'output_dir'
        if 'pack_model_dir' in d:
            md = MODELS / d
            od = MODELS / 'wave2_m0s7_s097' / 'output_dir'
        print('=' * 90)
        print(f'### {name}   model_dir={md}   output_dir={od}')
        if not md.exists() or not od.exists():
            print('  !! 源缺失, 跳过'); continue
        for line in harden_vendor.harden(md):
            print('  ' + line)
        r = subprocess.run([PY, str(TOOLS / 'package_generic.py'), '--name', name,
                            '--outdir', str(od), '--modeldir', str(md)],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=3600)
        print('  ' + (r.stdout or '').strip().replace('\n', '\n  '))
        if 'PACK_PASS' not in (r.stdout or ''):
            print('  !! 打包失败'); print((r.stderr or '')[-500:])
        else:
            print(f'  ✅ 完成 ({time.time():.0f})')
    print('REBUILD_DONE')


if __name__ == '__main__':
    main()
