# -*- coding: utf-8 -*-
"""runner 提速勘察 v2 (2026-09-13 夜, GPU 空闲后跑)
目标: 不改权重、不改出图内容, 只改【实现/精度配置】把 512 时延压下来。
关键怀疑: torch.autocast 会把 group_norm/layer_norm 等强制 fp32, VAE/解码器里 GroupNorm 极多
          -> 纯 fp16(关 autocast) 可能大幅提速。
输出: 组件耗时 + 各变体耗时 + 与基线的最大像素差(确认出图不变质)。
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
MODEL_DIR = r'D:/models/final_compliant/model_dir'
sys.path.insert(0, MODEL_DIR)
import torch
import torch.nn.functional as F
import importlib.util

spec = importlib.util.spec_from_file_location('m24runner', os.path.join(MODEL_DIR, 'runner.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dev = torch.device('cuda:0')
torch.manual_seed(0)
x = (torch.rand(1, 3, 512, 512, device=dev) * 2 - 1).half()

print('=== load Runner ===', flush=True)
t0 = time.perf_counter()
R = mod.Runner(MODEL_DIR)
print(f'load {time.perf_counter()-t0:.1f}s', flush=True)
M = R.model

def bench(fn, n=20, warm=5):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

# ---------- 基线 ----------
base = bench(lambda: R.infer(x))
print(f'[基线] R.infer = {base:.1f} ms  (加速比 {285.6/base:.3f}x)', flush=True)

# ---------- 参考输出 ----------
with torch.no_grad():
    ref = R.infer(x).float().clone()

# ---------- 组件/子模块 ----------
z = torch.randn(1, 4, 64, 64, device=dev, dtype=torch.float16)
zf = None
with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.float16):
        raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=M.dinov2_size, mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - M.mean) / M.std
        v = M.venc.prepare_tokens_with_masks(xn, None)
        for i, blk in enumerate(M.venc.blocks):
            v = blk(v)
            if i in M.layer_list:
                zf = v[:, 1:].detach()
    lat = None

def t_dino():
    with torch.autocast('cuda', dtype=torch.float16):
        raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=M.dinov2_size, mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - M.mean) / M.std
        v = M.venc.prepare_tokens_with_masks(xn, None)
        for i, blk in enumerate(M.venc.blocks):
            v = blk(v)
            if i in M.layer_list:
                return v[:, 1:].detach()

def t_vae():
    with torch.autocast('cuda', dtype=torch.float16):
        return M.vae.encode(x).latent_dist.sample() * M.scale

with torch.no_grad():
    lat0 = t_vae()
    tt = torch.ones(1, device=dev); tz = torch.zeros(1, device=dev)
    # autocast 下 latent_dist.sample() 里的 exp() 被提升到 fp32 -> 必须在 autocast 内进 DiT
    # (否则 conv 报 Input float vs bias Half)
    with torch.autocast('cuda', dtype=torch.float16):
        u0 = M.diT(torch.cat([lat0.half(), z], 1), tt, tz, [zf])

def t_dit():
    with torch.autocast('cuda', dtype=torch.float16):
        lat = M.vae.encode(x).latent_dist.sample() * M.scale
        return M.diT(torch.cat([lat, z], 1), tt, tz, [zf])

def t_dec():
    with torch.autocast('cuda', dtype=torch.float16):
        return M.dec((z - u0) / M.scale)

def t_enc_only():
    with torch.autocast('cuda', dtype=torch.float16):
        return M.vae.encoder(x)

comp = {'total': base, 'dinov2': bench(t_dino), 'vae_encode(sample)': bench(t_vae),
        'vae.encoder_only': bench(t_enc_only), 'dit(+vae)': bench(t_dit), 'dec_only': bench(t_dec)}
print('[组件 ms]', json.dumps({k: round(v,1) for k,v in comp.items()}), flush=True)

# ---------- 变体 ----------
variants = {}

def full_nocast():
    raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=M.dinov2_size, mode='bicubic').clamp(0., 255.)
    xn = (raw / 255.0 - M.mean) / M.std
    v = M.venc.prepare_tokens_with_masks(xn, None)
    zfeat = None
    for i, blk in enumerate(M.venc.blocks):
        v = blk(v)
        if i in M.layer_list:
            zfeat = v[:, 1:].detach()
    lat = M.vae.encode(x).latent_dist.sample() * M.scale
    u = M.diT(torch.cat([lat, z], 1), tt, tz, [zfeat])
    return M.dec((z - u) / M.scale).clamp(-1, 1)

for name, fn in [('pure_fp16_nocast', full_nocast)]:
    try:
        with torch.no_grad():
            o = fn()
        d = (o.float() - ref).abs().max().item()
        variants[name] = {'ms': bench(fn), 'maxdiff': d}
    except Exception as e:
        variants[name] = {'err': f'{type(e).__name__}: {str(e)[:140]}'}
    print(f'[{name}] {variants[name]}', flush=True)

# cudnn benchmark
try:
    torch.backends.cudnn.benchmark = True
    m = bench(lambda: R.infer(x))
    torch.backends.cudnn.benchmark = False
    variants['cudnn_benchmark'] = {'ms': m}
except Exception as e:
    variants['cudnn_benchmark'] = {'err': str(e)[:100]}
print('[cudnn_benchmark]', variants.get('cudnn_benchmark'), flush=True)

# compile 各子模块
for tag, attr in (('compile_dec', 'dec'), ('compile_diT', 'diT')):
    try:
        obj = getattr(M, attr.replace('compile_', ''))
        c = torch.compile(obj, mode='max-autotune', fullgraph=False)
        setattr(M, attr.replace('compile_', ''), c)
        with torch.no_grad():
            o = R.infer(x)
        d = (o.float() - ref).abs().max().item()
        variants[tag] = {'ms': bench(lambda: R.infer(x)), 'maxdiff': d}
    except Exception as e:
        variants[tag] = {'err': f'{type(e).__name__}: {str(e)[:140]}'}
    print(f'[{tag}] {variants[tag]}', flush=True)

res = {'baseline_ms': base, 'speedup': 285.6/base, 'components': comp, 'variants': variants}
json.dump(res, open(r'D:/tmp_timing/speed_probe_v2.json', 'w'), indent=1, ensure_ascii=False, default=str)
print('=== 汇总 ===')
print(json.dumps({k: (round(v[0] if isinstance(v, tuple) else v, 1) if isinstance(v, (int, float)) else v)
                  for k, v in {**comp, **{k: (vv.get('ms') if isinstance(vv, dict) else vv) for k, vv in variants.items()}}.items()},
                 ensure_ascii=False, indent=1, default=str))
print('SPEED_PROBE_V2_DONE', flush=True)
