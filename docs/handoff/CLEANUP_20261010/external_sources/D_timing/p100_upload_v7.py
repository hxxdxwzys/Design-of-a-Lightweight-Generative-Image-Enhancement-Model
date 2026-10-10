# -*- coding: utf-8 -*-
"""重传 v7 LoRA (2026-09-02): 修复截断, 校验字节数"""
import sys
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect  # noqa: E402

LOCAL = Path(r'D:/models/vosr_lora/v7/lora_s5000.pt')
SIZE = LOCAL.stat().st_size
REMOTE = 'p100/tools/v7/lora_s5000.pt'


def main():
    cli = connect()
    sftp = cli.open_sftp()
    try:
        if sftp.stat(REMOTE).st_size == SIZE:
            print('已存在且完整, 无需重传', flush=True)
            return
    except Exception:
        pass
    sftp.remove(REMOTE) if _exists(sftp) else None
    sftp.put(str(LOCAL), REMOTE)
    got = sftp.stat(REMOTE).st_size
    print(f'上传完成: {got} bytes, 期望 {SIZE}, {"OK" if got == SIZE else "BAD!"}', flush=True)


def _exists(sftp):
    try:
        sftp.stat(REMOTE)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    main()
