# -*- coding: utf-8 -*-
"""批量补齐 P100 训练存档到本地 (2026-09-02): 下载 + 202371666 字节校验"""
import sys
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect  # noqa: E402

SIZE = 202371666
base = Path(r'D:/tmp_timing/p100_ckpts')
ITEMS = [
    ('p768_w20', 's2000'), ('p768_w20', 's2500'),
    ('p768_w15', 's1000'), ('p768_w15', 's1500'),
    ('p768_w15', 's2000'), ('p768_w15', 's2500'),
]


def main():
    cli = connect()
    sftp = cli.open_sftp()
    for v, name in ITEMS:
        dst = base / f'{v}_lora_{name}.pt'
        if dst.exists() and dst.stat().st_size == SIZE:
            print(f'{v}_{name}: 已有 OK', flush=True)
            continue
        if dst.exists():
            dst.unlink()
        sftp.get(f'p100/runs/{v}/lora_{name}.pt', str(dst))
        ok = dst.stat().st_size == SIZE
        print(f'{v}_{name}: {dst.stat().st_size} bytes {"OK" if ok else "BAD!"}', flush=True)
    sftp.close()
    cli.close()
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
