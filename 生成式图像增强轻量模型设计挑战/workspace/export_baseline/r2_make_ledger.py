# -*- coding: utf-8 -*-
"""创建 EXPORT-01R2 新批次总账（关联旧账；旧+新不超硬上限）。

用法: python r2_make_ledger.py <new_ledger_path>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402

OLD_LEDGER = Path('C:/models/exports/EXPORT01R_20260917_150506/ledger.json')


def sha256_file(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    old = json.loads(OLD_LEDGER.read_text(encoding='utf-8'))
    prior = {'ledger': str(OLD_LEDGER), 'ledger_sha256': sha256_file(OLD_LEDGER),
             'used': old.get('used'), 'supervisor_wall_s': 71.083,
             'combined_caps': {'component': 48, 'wall_s': 1800.0}}
    led = r_ledger.Ledger(out, top=12, top4k=4, component=38, wall_s=1728.0, disk_gib=8.0,
                          prior=prior)
    # 立即落盘一次，写入 prior 关联
    led._write(led._read())
    print(json.dumps({'ledger': str(out), 'limits': led.limits, 'prior': prior['used']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
