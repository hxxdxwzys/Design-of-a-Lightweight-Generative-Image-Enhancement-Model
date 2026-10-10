# -*- coding: utf-8 -*-
"""验证 PixRestore 合规 runner: 与直接模型路径逐像素对比 + 时延"""
import sys, time, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch
from PIL import Image
from types import SimpleNamespace
import json
from safetensors.torch import load_file

MD = r'D:/models/pkg_PixB/model_dir'
sys.path.insert(0, MD)
import importlib.util
spec = importlib.util.spec_from_file_location('pixrunner', os.path.join(MD, 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t0 = time.time(); R = mod.Runner(MD)
print(f'[load] {time.time()-t0:.1f}s', flush=True)

# 与"直接路径"对比
sys.path.insert(0, r'D:/models/newsrc/PixRestore_code')
from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
from pixrestore.vision import extract_layers
CKPT = Path(r'D:/models/newsrc/PixRestore/pixrestore-b')
cfg = json.loads((CKPT/'config.json').read_text(encoding='utf-8'))
cfg['dinov2_repository'] = r'D:/models/newsrc/dinov2_repo'
cfg['dinov2_checkpoint'] = r'D:/models/newsrc/dinov2/dinov2_vitb14_pretrain.pth'
cfg = SimpleNamespace(**cfg)
dm = mod._build_dit(vars(cfg)).cuda().half().eval()
dm.load_state_dict(load_file(str(CKPT/'clean_weights'/'ema_model.safetensors')), strict=False)
fn = lambda x: (mod._features.__get__(R, mod.Runner))(x) if False else None
# 直接用 R 的 encoder 与 flow 复现"官方 repo"路径
from pixrestore.vision import preprocess_dinov2
x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1).half()
with torch.no_grad():
    fea_ref = extract_layers(R.venc, x, R.layers, R.enc_size)
    fea_my = R._features(x)
    dfeat = max(float((a - b).abs().max()) for a, b in zip(fea_ref, fea_my))
    print(f'[features] 与 repo 路径最大差 = {dfeat:.5f}', flush=True)
    with torch.autocast('cuda', dtype=torch.float16):
        o_ref = R.flow.sample_multistep_fm(dm, x, venc_fea=fea_ref, n_steps=1, schedule='linear')
    y = R.infer(x)
    print(f'[output] 与 repo 路径最大差 = {float((o_ref.float() - y).abs().max()):.5f}', flush=True)
    print(f'[out] shape={tuple(y.shape)} std={y.std().item():.3f} nan={bool(torch.isnan(y).any())}', flush=True)
    xc = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1)
    for _ in range(3):
        y = R.infer(xc)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        y = R.infer(xc)
    torch.cuda.synchronize()
ms = (time.perf_counter() - t) / 10 * 1000
print(f'[latency] {ms:.1f} ms -> {285.6/ms:.2f}x', flush=True)
print('PIX_VERIFY_DONE')
