# -*- coding: utf-8 -*-
"""C3 隔离预检（只作用于本任务新目录，不触碰共享目录与旧综合门禁）。

用法: python isolated_precheck.py <run_dir> <zip_path>
检查：
  1) 解包到独立临时目录，逐条核对包内文件集合/大小/哈希（对照 zip_record.json）；
  2) 纯 ASCII 路径、无可能遮蔽官方依赖的桩包目录；
  3) 官方解释器兼容审计：ast.parse(feature_version=(3,8)) 检查 model_dir 下全部 .py；
  4) 全新子进程：按路径导入 runner.py、torch.jit.load、512x512 前向一次并回报哈希。
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORBIDDEN = {'timm', 'einops', 'pyiqa', 'transformers', 'diffusers', 'peft'}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


CHILD = r'''
import hashlib, json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'model_dir'))
import importlib.util
spec = importlib.util.spec_from_file_location('delivery_runner_pkg', str(root / 'model_dir' / 'runner.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
r = mod.Runner(str(root / 'model_dir'))
n = 512
y = torch.linspace(-1.0, 1.0, n * n).reshape(1, 1, n, n).repeat(1, 3, 1, 1)
timing = {}
out = r.infer(y, prompt='', timing=timing)
print(json.dumps({
    'ok': True, 'shape': list(out.shape), 'dtype': str(out.dtype),
    'finite': bool(torch.isfinite(out).all()),
    'min': float(out.min()), 'max': float(out.max()),
    'sha256': hashlib.sha256(out.detach().cpu().float().numpy().tobytes()).hexdigest(),
    'timing': timing,
}, ensure_ascii=False))
'''


def main():
    run = Path(sys.argv[1]).resolve()
    zp = Path(sys.argv[2]).resolve()
    from disk_guard import require_free
    require_free(Path(tempfile.gettempdir()), 5.0, 'C3 隔离预检前水位（解包副本）')
    rec = {'schema': 'EXPORT01-isolated-precheck/1', 'zip': str(zp),
           'zip_sha256': sha256_file(zp), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix='export01_precheck_'))
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        z.extractall(tmp)
    rec['extract_dir'] = str(tmp)
    rec['n_entries'] = len(names)
    rec['all_ascii'] = all(n.isascii() for n in names)
    rec['has_model_dir'] = any(n.startswith('model_dir/') for n in names)
    rec['n_jpg'] = sum(1 for n in names if n.startswith('output_dir/') and n.endswith('.jpg'))
    mdl = tmp / 'model_dir'
    rec['forbidden_dirs'] = sorted({p.name for p in mdl.rglob('*')
                                    if p.is_dir() and p.name in FORBIDDEN})
    zrec = json.loads((run / 'evidence' / 'zip_record.json').read_text(encoding='utf-8'))
    hashes_ok = True
    for item in zrec['model_files']:
        p = tmp / item['path']
        if not p.is_file() or sha256_file(p) != item['sha256']:
            hashes_ok = False
    rec['model_file_hashes_match'] = bool(hashes_ok)
    # 官方解释器兼容：以 3.8 语法版本解析
    syntax = {}
    for p in sorted(mdl.rglob('*.py')):
        try:
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p), feature_version=(3, 8))
            syntax[p.name] = 'OK'
        except SyntaxError as e:
            syntax[p.name] = 'SYNTAX_ERROR: %s' % e
    rec['py38_syntax'] = syntax
    rec['py38_syntax_ok'] = all(v == 'OK' for v in syntax.values())
    # 遮蔽审计：官方 pyiqa 依赖不得被 model_dir 内同名包遮蔽
    rec['shadowing_risk'] = 'NONE' if not rec['forbidden_dirs'] else 'FOUND'
    # 全新子进程 512 前向
    child = tmp / 'check_forward.py'
    child.write_text(CHILD, encoding='utf-8')
    t = time.perf_counter()
    proc = subprocess.run([sys.executable, '-u', str(child), str(tmp)], cwd=str(tmp),
                          capture_output=True, text=True, timeout=900)
    rec['forward_exit_code'] = proc.returncode
    rec['forward_wall_s'] = round(time.perf_counter() - t, 3)
    rec['forward_stdout_tail'] = proc.stdout.strip().splitlines()[-1:] or []
    rec['forward_stderr_tail'] = proc.stderr.strip().splitlines()[-3:]
    try:
        rec['forward_result'] = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        rec['forward_result'] = None
    rec['wall_s'] = round(time.perf_counter() - t0, 3)
    rec['verdict'] = 'PASS' if (rec['all_ascii'] and rec['n_jpg'] == 100 and
                                rec['has_model_dir'] and hashes_ok and
                                rec['py38_syntax_ok'] and rec['shadowing_risk'] == 'NONE' and
                                proc.returncode == 0 and rec['forward_result'] and
                                rec['forward_result'].get('ok')) else 'FAIL'
    (run / 'evidence' / 'isolated_precheck.json').write_text(
        json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('verdict', 'n_entries', 'n_jpg', 'all_ascii',
                                          'model_file_hashes_match', 'py38_syntax_ok',
                                          'shadowing_risk', 'forward_exit_code', 'wall_s')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
