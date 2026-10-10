# -*- coding: utf-8 -*-
"""C3 打包：model_dir/ + output_dir/ -> 本地候选 ZIP（ASCII 名称，<=1e10 字节）。

用法: python make_zip.py <run_dir> --name Track2_EXPORT01_v1
产出: <run_dir>/<name>.zip 与 <run_dir>/evidence/zip_record.json（ZIP 哈希写在包外）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

FORBIDDEN_DIRS = {'timm', 'einops', 'pyiqa', 'transformers', 'diffusers', 'peft', 'torch',
                  'torchvision', 'numpy', 'cv2', 'PIL'}
MAX_BYTES = 10000000000


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    name = sys.argv[sys.argv.index('--name') + 1]
    if not name.isascii():
        raise SystemExit('ZIP 名称必须为纯 ASCII: %r' % name)
    from disk_guard import require_free
    require_free(run, 6.0, 'C3 打包前水位（ZIP + 解包副本）')
    mdl, outd = run / 'model_dir', run / 'output_dir'
    if not mdl.is_dir() or not outd.is_dir():
        raise SystemExit('缺少 model_dir/ 或 output_dir/')
    files = []
    for p in sorted(mdl.rglob('*')):
        if p.is_file():
            files.append((p, Path('model_dir') / p.relative_to(mdl)))
        elif p.is_dir() and p.name in FORBIDDEN_DIRS:
            raise SystemExit('model_dir 内出现可能遮蔽官方依赖的目录: %s' % p)
    jpgs = sorted(outd.glob('case*.jpg'), key=lambda q: int(q.stem[4:]))
    if len(jpgs) != 100:
        raise SystemExit('output_dir 需要恰好 100 张 caseN.jpg，实为 %d' % len(jpgs))
    for p in jpgs:
        files.append((p, Path('output_dir') / p.name))
    for _, rel in files:
        if not rel.as_posix().isascii():
            raise SystemExit('包内路径必须为纯 ASCII: %s' % rel)
    zp = run / ('%s.zip' % name)
    t0 = time.perf_counter()
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, rel in files:
            z.write(src, rel.as_posix())
    entries = []
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            entries.append({'path': info.filename, 'bytes': info.file_size,
                            'crc32': '%08x' % info.CRC})
    rec = {
        'schema': 'EXPORT01-zip-record/1',
        'zip_path': str(zp), 'zip_name': zp.name,
        'zip_bytes': zp.stat().st_size, 'zip_sha256': sha256_file(zp),
        'max_bytes': MAX_BYTES, 'within_limit': zp.stat().st_size <= MAX_BYTES,
        'n_entries': len(entries), 'n_jpg': len(jpgs), 'n_model_files': len(files) - len(jpgs),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pack_s': round(time.perf_counter() - t0, 3),
        'model_files': [{'path': e[1].as_posix(), 'bytes': e[0].stat().st_size,
                         'sha256': sha256_file(e[0])} for e in files if e[1].parts[0] == 'model_dir'],
        'entry_paths_sha256': hashlib.sha256(
            '\n'.join(e['path'] for e in entries).encode('utf-8')).hexdigest(),
    }
    (run / 'evidence' / 'zip_record.json').write_text(
        json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('zip_name', 'zip_bytes', 'zip_sha256', 'within_limit',
                                          'n_entries', 'n_jpg', 'n_model_files')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
