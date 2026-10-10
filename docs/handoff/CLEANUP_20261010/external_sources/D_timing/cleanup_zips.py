# -*- coding: utf-8 -*-
"""zip/目录 清理 (2026-09-11)
干跑: python cleanup_zips.py            (只打印)
执行: python cleanup_zips.py --apply
原则: ① 先归档证据(旧 runner.py)再删大件 ② 硬链接副本删除不损字节 ③ 可再生的渲染中间产物可删
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

APPLY = '--apply' in sys.argv
MODELS = Path(r'D:/models')
HIST = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
ARCH = HIST / '_失败原件_runner快照'

# ① 归档: 失败四发的旧 runner.py (证据, ~24KB)
ARCHIVE = [
    (HIST / 'Track2_CompliantM24——.zip', 'runner_M24_v1_FAILED.py'),
    (HIST / 'Track2_14B_M0S1_SC093——.zip', 'runner_M0S1_SC093_v1_FAILED.py'),
    (HIST / 'Track2_14B_M0S7_SC10——.zip', 'runner_M0S7_SC10_v1_FAILED.py'),
    (HIST / 'Track2_14B_G3W18S1_SC093——.zip', 'runner_G3W18S1_v1_FAILED.py'),
]

# ② 删除: (路径, 说明)
DELETE = [
    *[(MODELS / f, '与 upload/ 硬链接的重复名 (删它释放 0 字节, 纯降噪)') for f in (
        'Track2_CompliantM24_v2.zip', 'Track2_14B_M0S1_SC093_v2.zip',
        'Track2_14B_M0S7_SC10_v2.zip', 'Track2_14B_M0S7_SC097.zip',
        'Track2_14B_G3W18S1_SC093_v2.zip')],
    *[(HIST / f, '失败四发原件 (已被 _v2 取代, runner 快照已归档)') for f in (
        'Track2_CompliantM24——.zip', 'Track2_14B_M0S1_SC093——.zip',
        'Track2_14B_M0S7_SC10——.zip', 'Track2_14B_G3W18S1_SC093——.zip')],
    *[(MODELS / d / 'render_png', '无损母版 png (提交用 jpg 已生成, 可重渲)') for d in (
        'wave2_m0s7_s097', 'wave2_m0s1_s097', 'wave2_m0s1_s090', 'wave2_m0s1_s100')],
    (MODELS / '_harness_zip_sim', '门禁临时解压目录 (每次跑门禁自动重建)'),
    (MODELS / 'wave2_m0s7_s097' / 'model_dir',
     '合并权重裸副本 (pack_model_dir 里已有同一份, 且 7s 可由 remerge 重建)'),
]


def size_of(p: Path):
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    return 0


def main():
    print('=' * 96)
    print('① 归档证据 (失败四发的旧 runner.py)')
    print('=' * 96)
    if not APPLY:
        for zp, name in ARCHIVE:
            print(f'  [干跑] {zp.name}  ->  {ARCH/name}')
    else:
        ARCH.mkdir(parents=True, exist_ok=True)
        for zp, name in ARCHIVE:
            if not zp.exists():
                print(f'  !! 缺 {zp.name}'); continue
            with zipfile.ZipFile(zp) as z:
                data = z.read('model_dir/runner.py')
            (ARCH / name).write_bytes(data)
            print(f'  ✅ {name}  ({len(data)} B)')

    print()
    print('=' * 96)
    print('② 删除')
    print('=' * 96)
    total = 0
    for p, why in DELETE:
        if not p.exists():
            print(f'  -- 不存在, 跳过: {p}')
            continue
        sz = size_of(p)
        total += sz
        print(f'  {"✅ 已删" if APPLY else "[干跑]"} {sz/2**30:6.2f} GiB  {p}')
        print(f'          理由: {why}')
        if APPLY:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    print()
    print(f'合计可释放: {total/2**30:.2f} GiB   {"(已执行)" if APPLY else "(干跑, 加 --apply 执行)"}')


if __name__ == '__main__':
    main()
