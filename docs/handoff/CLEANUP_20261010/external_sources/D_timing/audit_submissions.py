# -*- coding: utf-8 -*-
"""提交件全量审计: D:/models 的 zip vs 提交历史 的归档 zip (内部文件指纹比对)"""
import hashlib
import zipfile
from pathlib import Path

TARGETS = ['model_dir/your_model.pt', 'model_dir/runner.py', 'output_dir/case1.jpg', 'output_dir/case50.jpg']


def fingerprint(path):
    try:
        z = zipfile.ZipFile(path)
        names = set(z.namelist())
        parts = []
        for t in TARGETS:
            if t in names:
                parts.append(hashlib.md5(z.read(t)).hexdigest()[:10])
            else:
                parts.append('MISSING')
        return '|'.join(parts)
    except Exception as e:
        return 'ERR:' + str(e)[:60]


def main():
    hist_dir = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
    arch = {}
    for p in sorted(hist_dir.glob('*.zip')):
        arch[p.name] = fingerprint(p)
        print('[arch] %s -> %s' % (p.name, arch[p.name]), flush=True)

    print('', flush=True)
    models_dir = Path(r'D:/models')
    for p in sorted(models_dir.glob('*.zip')):
        fp = fingerprint(p)
        matches = [n for n, f in arch.items() if f == fp]
        tag = ('[ARCHIVED] ' + ', '.join(matches)) if matches else '[NOT-ARCHIVED]'
        print('[models] %s (%.0fMB) -> %s %s' % (p.name, p.stat().st_size / 1e6, fp, tag), flush=True)


if __name__ == '__main__':
    main()
