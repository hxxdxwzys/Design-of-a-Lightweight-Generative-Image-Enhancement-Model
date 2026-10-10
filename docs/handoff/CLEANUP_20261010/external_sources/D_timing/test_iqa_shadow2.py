# -*- coding: utf-8 -*-
"""判定 model_dir 遮蔽是否会导致官方 IQAEvaluator 加载失败。

官方质量指标: CLIPIQA / MUSIQ / MANIQA / NIQE (均由 pyiqa 提供, 依赖 timm/einops)。
两种环境各测一遍:
  A 干净环境      (不把我们 model_dir 放进 sys.path)  -> 基线
  B 遮蔽环境      (先 import 我们的 runner, 再加载 IQA) -> 复现官方场景
用法: D:/Python/python.exe test_iqa_shadow2.py A|B
"""
from __future__ import annotations

import os
import sys

MD = r'D:/models/final_compliant_14b/model_dir'
METRICS = ['clipiqa', 'musiq', 'maniqa', 'niqe']


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    os.chdir(r'C:/models/finals')
    print(f'### 模式 {mode}: {"干净环境" if mode == "A" else "遮蔽环境(先 import runner)"}')
    print('python', sys.version.split()[0])

    if mode == 'B':
        sys.path.insert(0, MD)
        import runner  # noqa: F401
        import timm
        import einops
        print('  timm   ->', os.path.abspath(timm.__file__))
        print('  einops ->', os.path.abspath(einops.__file__))
        print('  sys.path[0] =', sys.path[0])
    else:
        import timm
        print('  (基线) timm ->', os.path.abspath(timm.__file__))

    import pyiqa
    print('  pyiqa', pyiqa.__version__)
    ok, bad = [], []
    for name in METRICS:
        try:
            m = pyiqa.create_metric(name, device='cpu')
            ok.append(name)
            print(f'  ✅ {name:8s} 加载 OK -> {type(m).__name__}')
        except Exception as e:
            bad.append((name, f'{type(e).__name__}: {e}'))
            print(f'  ❌ {name:8s} 加载失败 -> {type(e).__name__}: {str(e)[:200]}')
    print(f'--- 模式 {mode} 结果: OK={ok} FAIL={[b[0] for b in bad]}')
    if bad:
        print('  失败详情:')
        for n, e in bad:
            print(f'    {n}: {e[:300]}')


if __name__ == '__main__':
    main()
