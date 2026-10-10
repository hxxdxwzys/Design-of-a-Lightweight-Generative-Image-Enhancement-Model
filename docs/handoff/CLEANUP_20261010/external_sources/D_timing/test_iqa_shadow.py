# -*- coding: utf-8 -*-
"""验证官方 IQAEvaluator 失败的一种可能机制: 我们的 model_dir 遮蔽了 pyiqa 的依赖。

背景: 2026-09-11 官方返回
  "Evaluation failed during quality: Failed to load IQAEvaluator, please contact committee to retry"
官方 IQA 评测 (CLIPIQA/MUSIQ/MANIQA/NIQE) 依赖 pyiqa -> 依赖 timm / einops。
我们的 model_dir 自带 timm/ einops/ diffusers/ fairscale/ 等同名包, 且 runner.py 用
sys.path.insert(0, _MODEL_DIR) 把 model_dir 插到最前面 -> 可能遮蔽官方环境里的真包。

本脚本对比"导入我们的 runner 之前/之后", 这些名字解析到哪个位置, 并尝试真正加载 clipiqa。
"""
from __future__ import annotations

import os
import sys

MD = r'D:/models/final_compliant_14b/model_dir'
NAMES = ('timm', 'einops', 'diffusers', 'fairscale', 'models', 'dinov2', 'vae_shim')


def probe(tag):
    print(f'--- {tag} ---')
    for name in NAMES:
        if name in sys.modules:
            m = sys.modules[name]
            f = os.path.abspath(getattr(m, '__file__', '') or '<namespace>')
            where = 'OURS(model_dir)' if os.path.abspath(MD) in f else 'external'
            print(f'  {name:10s} [已在 sys.modules] -> {where}  {f}')
        else:
            print(f'  {name:10s} [未导入]')


def main():
    print('python', sys.version.split()[0])
    os.chdir(r'C:/models/finals')  # runner 不依赖 cwd, 保险起见
    sys.path.insert(0, MD)

    # pyiqa 的真实依赖在导入我们 runner 之前解析成什么
    import timm as _t
    import einops as _e
    print('=== 对照组: 未导入 runner 时 ===')
    print('  timm   ->', os.path.abspath(_t.__file__))
    print('  einops ->', os.path.abspath(_e.__file__))
    print('  pyiqa 版本:', __import__('pyiqa').__version__)

    print()
    print('=== 导入我们的 runner (模拟官方 harness 先加载 runner) ===')
    import runner  # noqa: F401
    print('  runner 导入完成, sys.path[0] =', sys.path[0])
    probe('导入 runner 之后')

    print()
    print('=== 尝试加载 IQA 评测器 (官方报错点) ===')
    try:
        import pyiqa
        m = pyiqa.create_metric('clipiqa', device='cpu')
        print('  clipiqa 加载 OK ->', type(m).__name__)
    except Exception as e:
        print(f'  ❌ clipiqa 加载失败: {type(e).__name__}: {e}')

    print()
    print('=== sys.path 前 5 项 ===')
    for i, p in enumerate(sys.path[:5]):
        print(f'  [{i}] {p}')


if __name__ == '__main__':
    main()
