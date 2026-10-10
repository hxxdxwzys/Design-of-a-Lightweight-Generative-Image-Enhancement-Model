# -*- coding: utf-8 -*-
"""对给定 model_dir 目录做兼容性冒烟: 插 sys.path -> import runner -> 512 infer (2026-09-11)
与 harness_sim_zip.py 同口径, 但不解压 zip (用于加固后的目录直接验证)。
用法: python smoke_runner_dir.py <model_dir> [more...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.chdir(r'C:/models/finals')      # dinov2 离线缓存判定依赖 cwd (见 HANDOFF 16.84 渲染链 bug)


def main():
    import torch
    ok_all = True
    for a in sys.argv[1:]:
        md = Path(a).resolve()
        print(f'===== {md}')
        if not (md / 'runner.py').exists():
            print('  !! 无 runner.py'); ok_all = False; continue
        sys.path.insert(0, str(md))
        for m in list(sys.modules):
            if m == 'runner' or m.split('.')[0] in ('models', 'dinov2', 'vae_shim',
                                                    'timm', 'einops', 'diffusers', 'fairscale'):
                del sys.modules[m]
        try:
            t0 = time.time()
            import runner as R
            print(f'  [1] import OK ({time.time()-t0:.1f}s)')
            r = R.Runner(str(md))
            x = torch.randn(1, 3, 512, 512)
            t1 = time.time()
            with torch.no_grad():
                y = r.infer(x)
            dt = time.time() - t1
            std = y.std().item()
            print(f'  [2] infer OK {tuple(y.shape)} range=({y.min().item():.2f},'
                  f'{y.max().item():.2f}) std={std:.3f} {dt:.1f}s')
            good = torch.isfinite(y).all() and std > 0.05
            print(f'  ==> {"SMOKE_PASS" if good else "SMOKE_FAIL(输出异常)"}')
            ok_all = ok_all and good
        except Exception as e:
            import traceback
            print(f'  ==> SMOKE_FAIL: {type(e).__name__}: {e}')
            print('     ' + traceback.format_exc().strip().splitlines()[-1][:200])
            ok_all = False
        finally:
            try:
                sys.path.remove(str(md))
            except ValueError:
                pass
        print()
    print('SMOKE_ALL_PASS' if ok_all else 'SMOKE_ANY_FAIL')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
