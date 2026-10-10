# -*- coding: utf-8 -*-
"""通过代理打通 MANIQA (2026-09-11): 设代理 -> 让 pyiqa 下载 timm backbone -> 冒烟"""
from __future__ import annotations

import os
import sys
import time

PROXY = 'http://127.0.0.1:7890'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = PROXY
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# HF 镜像也可用; 先直连(走代理)
os.environ.pop('HF_ENDPOINT', None)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print('代理:', PROXY, flush=True)

# 0) 连通性测试
try:
    import httpx
    r = httpx.head('https://huggingface.co', timeout=20, follow_redirects=True)
    print('HF 连通性:', r.status_code, flush=True)
except Exception as e:
    print('HF 连通性失败:', type(e).__name__, str(e)[:200], flush=True)

# 1) 直接让 timm 拉 vit_base_patch8_224
t0 = time.time()
try:
    import timm
    m = timm.create_model('vit_base_patch8_224', pretrained=True)
    n = sum(p.numel() for p in m.parameters())
    print(f'timm vit_base_patch8_224 OK: {n/1e6:.1f}M params ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    import traceback
    print('timm 拉取失败:', type(e).__name__, flush=True)
    print(traceback.format_exc()[-1200:], flush=True)

# 2) pyiqa MANIQA
t0 = time.time()
try:
    import torch
    import pyiqa
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = pyiqa.create_metric('maniqa', device=dev)
    print(f'MANIQA 创建 OK ({time.time()-t0:.0f}s, device={dev})', flush=True)
    x = torch.rand(1, 3, 512, 512, device=dev)
    with torch.no_grad():
        v = metric(x)
    print('MANIQA 冒烟输出:', float(v.item()), flush=True)
    print('MANIQA_READY', flush=True)
except Exception as e:
    import traceback
    print('MANIQA 失败:', type(e).__name__, flush=True)
    print(traceback.format_exc()[-1500:], flush=True)
