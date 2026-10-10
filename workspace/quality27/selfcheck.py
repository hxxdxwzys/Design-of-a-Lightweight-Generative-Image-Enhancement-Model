"""Portable, fail-closed identity check for the separately labelled V4t1 edge revision."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
MODEL='2770215987e2b5585dc26870a8a6fee05672bbf514267336498a1bf9e6409b62'
RUNNER='8e434e0240eed20b924e04f5f7338a037c8ec7821b8067159fe50ac7492ff70b'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def verify_identity(root=HERE):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    required={'infer.py','selfcheck.py','model_dir/your_model.pt','model_dir/runner.py','requirements.txt'}
    if not required.issubset(m['files']):raise ValueError('Incomplete manifest')
    if m['version']!='V4t1_EdgeFix_20260927':raise ValueError('Wrong version')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=set(m['files']):raise ValueError('File set differs from manifest')
    for rel,e in m['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or p==root:raise ValueError('Unsafe path')
        if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('Changed file: '+rel)
    if sha(root/'model_dir/your_model.pt')!=MODEL or sha(root/'model_dir/runner.py')!=RUNNER:raise ValueError('Not V4t1 parent')
    return m


if __name__=='__main__':
    verify_identity();print('IDENTITY_PASS; no GPU forward performed')
