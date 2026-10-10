# -*- coding: utf-8 -*-
"""第二轮清理 (2026-09-11, 用户批准): 只看 upload/ 作为上传源
删: ① D:/models 顶层的 5 个 zip 名(与 upload/ 硬链接, 释放 0 字节, 纯降噪)
    ② 老世界产物目录 (cnn_out_* / final_graph* / final_dryrun_out / final_quality_runner)
    ③ --harness-sim 时额外删 _harness_zip_sim (门禁临时目录, 会自动重建)
保留: upload/ 5 包 | final_compliant* 与 wave2_* 打包源 | 权重资产(vosr_lora/sd21base/open_clip/finals/hypir)
用法: python cleanup_round2.py [--apply] [--harness-sim]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

APPLY = '--apply' in sys.argv
WITH_SIM = '--harness-sim' in sys.argv
M = Path(r'D:/models')

ROOT_ZIPS = ['Track2_M24_v3.zip', 'Track2_M0S1_SC093_v3.zip', 'Track2_M0S7_SC10_v3.zip',
             'Track2_G3W18S1_SC093_v3.zip', 'Track2_M0S7_SC097_v3.zip']
OLD_DIRS = ['cnn_out_f2k', 'cnn_out_mix_m26', 'cnn_out_mix_s7_m16', 'cnn_out_mix_s7_m17',
            'cnn_out_s6k', 'final_graph', 'final_graph7', 'final_dryrun_out',
            'final_quality_runner']


def size_of(p: Path):
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    return 0


def main():
    items = [(M / n, '顶层 zip 名 (与 upload/ 硬链接, 0 字节)') for n in ROOT_ZIPS]
    items += [(M / d, '老世界产物目录 (已作废)') for d in OLD_DIRS]
    if WITH_SIM:
        items.append((M / '_harness_zip_sim', '门禁临时解压目录 (自动重建)'))

    total = 0
    freed = 0
    for p, why in items:
        if not p.exists():
            print(f'  -- 不存在, 跳过: {p.name}')
            continue
        sz = size_of(p)
        total += sz
        real = 0 if (p.is_file() and p.stat().st_nlink > 1) else sz
        freed += real
        print(f'  {"已删" if APPLY else "[干跑]"} {sz/2**30:6.2f} GiB (实释放 {real/2**30:5.2f})  {p.name}')
        print(f'         理由: {why}')
        if APPLY:
            p.unlink() if p.is_file() else shutil.rmtree(p)
    print(f'\n合计涉及 {total/2**30:.2f} GiB, 实际释放约 {freed/2**30:.2f} GiB'
          f'   {"(已执行)" if APPLY else "(干跑, 加 --apply)"}')
    print('\n=== upload/ 最终上传源（唯一）===')
    for f in sorted((M / 'upload').glob('*.zip')):
        print(f'  {f.stat().st_size/2**30:.2f} GiB  {f.name}')


if __name__ == '__main__':
    main()
