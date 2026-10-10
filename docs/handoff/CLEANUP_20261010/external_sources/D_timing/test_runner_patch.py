# -*- coding: utf-8 -*-
"""验证 runner.py 补丁是否解除了对官方 IQA 依赖的遮蔽 (2026-09-11)

期望 (补丁后):
  timm / einops 解析到 site-packages (真包), 不再是 model_dir 里的裁剪桩
  from timm.models.vision_transformer import Block  -> 成功
  pyiqa maniqa -> 不再报 ImportError: cannot import name 'Block'
                  (剩下的是网络类错误, 属平台侧)
  并且我们自己的模型模块仍可用 (LightningDiT / dinov2 / vae_shim 正常导入)
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MD = sys.argv[1] if len(sys.argv) > 1 else r'D:/models/final_compliant_14b/model_dir'
MD_ABS = os.path.abspath(MD)
os.chdir(r'C:/models/finals')
sys.path.insert(0, MD)

print(f'### 被测 model_dir: {MD}')
import runner  # noqa: F401,E402

print('  sys.path[0]      =', sys.path[0])
print('  model_dir 在 path =', MD in sys.path, '| 位置', sys.path.index(MD) if MD in sys.path else None)

import timm  # noqa: E402
import einops  # noqa: E402

for name, mod in (('timm', timm), ('einops', einops)):
    f = os.path.abspath(getattr(mod, '__file__', '') or '')
    print(f'  {name:8s} -> {"❌ 仍是桩包" if f.startswith(MD_ABS) else "✅ 真包"}  {f}')

print('\n### 官方 IQA 会走的 import 路径')
try:
    from timm.models.vision_transformer import Block  # noqa: F401
    print('  ✅ from timm.models.vision_transformer import Block  OK')
except Exception as e:
    print(f'  ❌ Block 导入失败: {type(e).__name__}: {e}')

try:
    import torch
    from einops import repeat
    out = repeat(torch.zeros(2, 3), '... n -> ... (n r)', r=2)
    print(f'  ✅ real einops repeat 可用, 形状 {tuple(out.shape)}')
except Exception as e:
    print(f'  ❌ einops repeat 失败: {type(e).__name__}: {e}')

print('\n### 逐个加载官方四个 IQA 指标')
import pyiqa  # noqa: E402
print('  pyiqa', pyiqa.__version__)
for name in ('clipiqa', 'musiq', 'maniqa', 'niqe'):
    try:
        pyiqa.create_metric(name, device='cpu')
        print(f'  ✅ {name:8s} 加载 OK')
    except Exception as e:
        msg = f'{type(e).__name__}: {str(e)[:120]}'
        shadow = 'Block' in str(e) or 'timm' in str(e) or 'einops' in str(e)
        print(f'  ⚠️ {name:8s} {msg}   {"[仍是遮蔽!]" if shadow else "[非遮蔽: 网络/权重类]"}')

print('\n### 我们自己的模块是否仍可用')
print('  LightningDiT     =', runner.LightningDiT.__name__)
print('  _dvt (dinov2)    =', runner._dvt.__name__)
print('  AutoencoderKL... =', runner.AutoencoderKLQwenImage2D.__name__)
