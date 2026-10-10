# -*- coding: utf-8 -*-
"""runner 提速空间勘察 (2026-09-13)
目标: 在不改权重、不改出图的前提下, 把 512 时延压下来 —— 这是唯一能合法越过 1x 的杠杆。
做法: 逐组件计时 + 逐个变体试验(纯 fp16 无 autocast / inference_mode / cudnn.benchmark /
      torch.compile / SDPA 替换 dinov2 注意力 / 只读诊断每层的 dtype)。
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
MODEL_DIR = r'D:/models/final_compliant/model_dir'
sys.path.insert(0, MODEL_DIR)
import torch

import importlib.util
spec = importlib.util.spec_from_file_location('m24runner', os.path.join(MODEL_DIR, 'runner.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dev = torch.device('cuda:0')
torch.manual_seed(0)
x = (torch.rand(1, 3, 512, 512, device=dev) * 2 - 1).half()

print('=== 构建 Runner ===', flush=True)
t0 = time.perf_counter()
R = mod.Runner(MODEL_DIR)
print(f'load {time.perf_counter()-t0:.1f}s', flush=True)
M = R.model
print('dtypes:', {n: str(p.dtype) for n, p in list(M.named_parameters())[:1]},
      '| vae:', next(M.vae.parameters()).dtype, '| dec:', next(M.dec.parameters()).dtype,
      '| diT:', next(M.diT.parameters()).dtype, '| venc:', next(M.venc.parameters()).dtype, flush=True)

def timeit(fn, n=10, warm=3):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

# --- 组件计时 (用原始 autocast 路径) ---
import torch.nn.functional as F
z = torch.randn(1, 4, 64, 64, device=dev, dtype=torch.float16)

def part_dino():
    with torch.autocast('cuda', dtype=torch.float16):
        raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=M.dinov2_size, mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - M.mean) / M.std
        v = M.venc.prepare_tokens_with_masks(xn, None)
        for i, blk in enumerate(M.venc.blocks):
            v = blk(v)
            if i in M.layer_list:
                return v[:, 1:].detach()

def part_vae():
    with torch.autocast('cuda', dtype=torch.float16):
        return M.vae.encode(x).latent_dist.sample() * M.scale

def part_dit(zf):
    with torch.autocast('cuda', dtype=torch.float16):
        lat = M.vae.encode(x).latent_dist.sample() * M.scale
        return M.diT(torch.cat([lat, z], 1), torch.ones(1, device=dev), torch.zeros(1, device=dev), [zf])

def part_dec(zfeat):
    with torch.autocast('cuda', dtype=torch.float16):
        lat = M.vae.encode(x).latent_dist.sample() * M.scale
        u = M.diT(torch.cat([lat, z], 1), torch.ones(1, device=dev), torch.zeros(1, device=dev), [zfeat])
        return M.dec((z - u) / M.scale).clamp(-1, 1)

zf = part_dino()
res = {}
res['total_infer'] = timeit(lambda: R.infer(x))
res['dinov2'] = timeit(part_dino)
res['vae_encode'] = timeit(part_vae)
res['dit(+vae)'] = timeit(lambda: part_dit(zf))
res['dec(+dit+vae)'] = timeit(lambda: part_dec(zf))
print('--- 组件 (ms) ---', json.dumps({k: round(v,1) for k,v in res.items()}), flush=True)

# --- 变体 A: 纯 fp16, 无 autocast ---
def full_nocast():
    raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=M.dinov2_size, mode='bicubic').clamp(0., 255.)
    xn = (raw / 255.0 - M.mean) / M.std
    v = M.venc.prepare_tokens_with_masks(xn, None)
    for i, blk in enumerate(M.venc.blocks):
        v = blk(v)
        if i in M.layer_list:
            zfeat = v[:, 1:].detach()
    lat = M.vae.encode(x).latent_dist.sample() * M.scale
    u = M.diT(torch.cat([lat, z], 1), torch.ones(1, device=dev), torch.zeros(1, device=dev), [zfeat])
    return M.dec((z - u) / M.scale).clamp(-1, 1)

try:
    res['nocast_fp16'] = timeit(full_nocast)
except Exception as e:
    res['nocast_fp16'] = f'ERR {type(e).__name__}: {str(e)[:120]}'

# --- 变体 B: inference_mode ---
try:
    with torch.inference_mode():
        for _ in range(3):
            full_nocast()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            full_nocast()
        torch.cuda.synchronize()
    res['inference_mode'] = (time.perf_counter() - t) / 10 * 1000
except Exception as e:
    res['inference_mode'] = f'ERR {type(e).__name__}: {str(e)[:120]}'

# --- 变体 C: cudnn.benchmark ---
try:
    torch.backends.cudnn.benchmark = True
    res['cudnn_benchmark'] = timeit(lambda: R.infer(x))
    torch.backends.cudnn.benchmark = False
except Exception as e:
    res['cudnn_benchmark'] = f'ERR {e}'

print('--- 变体 (ms) ---', json.dumps({k: (round(v,1) if isinstance(v,float) else v) for k,v in res.items()}, ensure_ascii=False), flush=True)
json.dump(res, open(r'D:/tmp_timing/speed_probe_components.json','w'), indent=1, ensure_ascii=False)
print('SPEED_PROBE_DONE', flush=True)
