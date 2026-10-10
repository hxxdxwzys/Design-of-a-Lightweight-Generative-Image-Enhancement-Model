# -*- coding: utf-8 -*-
"""官方格式严格审计 (2026-09-11)
逐条对照官方要求（赛题 txt L36–55 + 官方样例 runner.py）：
  1. 顶层恰为 output_dir/ + model_dir/
  2. output_dir 内恰 100 个 .jpg，名字与官方测试集【完全一致】
  3. model_dir 内含 your_model.pt 与 runner.py
  4. 无 __pycache__/.pyc 等杂质
  5. 接口签名与官方样例一致: Runner(model_dir) / infer(image_tensor, prompt="")
  6. 实跑 512×512 [-1,1] 输入 -> [1,3,512,512] 输出, 有限且非退化
用法: python audit_official_format.py <zip> [more.zip ...]
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TESTDIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PY = r'D:/Python/python.exe'
SIM = Path(r'D:/models/_harness_zip_sim')

SMOKE = r'''
import inspect, os, sys
import torch
md = os.path.abspath(sys.argv[1])
sys.path.insert(0, md)
import runner as R
sig_init = str(inspect.signature(R.Runner.__init__))
sig_infer = str(inspect.signature(R.Runner.infer))
print('SIG init ', sig_init)
print('SIG infer', sig_infer)
obj = R.Runner(md)
x = torch.rand(1, 3, 512, 512) * 2 - 1          # [-1,1] fp32, 与官方 harness 同约定
with torch.no_grad():
    y = obj.infer(x)
print('OUT shape=%s dtype=%s min=%.3f max=%.3f std=%.3f' % (
    tuple(y.shape), y.dtype, y.min().item(), y.max().item(), y.std().item()))
ok = (tuple(y.shape) == (1, 3, 512, 512) and torch.isfinite(y).all()
      and y.std().item() > 0.05 and y.min().item() >= -1.01 and y.max().item() <= 1.01)
print('IFACE_' + ('PASS' if ok else 'FAIL'))
'''


def main():
    zips = [Path(a) for a in sys.argv[1:]]
    if not zips:
        zips = sorted(Path(r'D:/models/upload').glob('*.zip'))
    official = {p.name for p in TESTDIR.glob('*')} if TESTDIR.exists() else set()
    print(f'官方测试集文件名数: {len(official)}')
    allok = True
    for zp in zips:
        print('=' * 84)
        print(f'审计: {zp.name}')
        ok = True
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        tops = sorted({n.split('/')[0] for n in names})
        c1 = tops == ['model_dir', 'output_dir']
        print(f'  {"✅" if c1 else "❌"} ① 顶层恰为 output_dir + model_dir: {tops}')
        ok &= c1

        jpgs = [n.split('/', 1)[1] for n in names
                if n.startswith('output_dir/') and n.lower().endswith('.jpg')]
        others = [n for n in names if n.startswith('output_dir/') and not n.lower().endswith('.jpg')]
        c2 = len(jpgs) == 100 and set(jpgs) == official and not others
        print(f'  {"✅" if c2 else "❌"} ② output_dir: {len(jpgs)} 个 jpg, '
              f'与官方测试集名字集合一致={set(jpgs) == official}, 杂质={len(others)}')
        ok &= c2

        mds = [n.split('/', 1)[1] for n in names if n.startswith('model_dir/')]
        c3 = 'your_model.pt' in mds and 'runner.py' in mds
        print(f'  {"✅" if c3 else "❌"} ③ model_dir 含 your_model.pt + runner.py '
              f'(共 {len(mds)} 项, 额外 {len(mds) - 2} 项)')
        ok &= c3

        junk = [n for n in names if '__pycache__' in n or n.endswith('.pyc')]
        c4 = not junk
        print(f'  {"✅" if c4 else "❌"} ④ 无 __pycache__/.pyc 杂质: {len(junk)} 个')
        ok &= c4

        # ⑤⑥ 解压跑接口
        import shutil
        if SIM.exists():
            shutil.rmtree(SIM)
        SIM.mkdir(parents=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(SIM)
        probe = Path(r'D:/tmp_timing/_probe_iface.py')
        probe.write_text(SMOKE, encoding='utf-8')
        r = subprocess.run([PY, str(probe), str(SIM / 'model_dir')], capture_output=True,
                           text=True, encoding='utf-8', errors='replace',
                           timeout=1800, cwd=r'C:/models/finals')
        out = (r.stdout or '') + (r.stderr or '')
        for ln in out.splitlines():
            if ln.startswith(('SIG', 'OUT', 'IFACE')):
                print('     ' + ln)
        c5 = 'SIG init  (self, model_dir' in out or 'SIG init  (self, model_dir:' in out
        c6 = 'IFACE_PASS' in out
        print(f'  {"✅" if c5 else "❌"} ⑤ 接口签名与官方一致 Runner(model_dir)')
        print(f'  {"✅" if c6 else "❌"} ⑥ 512×512 实跑 [1,3,H,W]∈[-1,1] -> 同尺寸')
        ok &= (c5 and c6)
        print(f'  ==> {"FORMAT_PASS" if ok else "FORMAT_FAIL"}')
        allok &= ok
    print('=' * 84)
    print('FORMAT_ALL_PASS' if allok else 'FORMAT_ANY_FAIL')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
