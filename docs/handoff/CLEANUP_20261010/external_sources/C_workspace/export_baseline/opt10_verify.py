# -*- coding: utf-8 -*-
"""OPT-10 验证：在真实运行器里确认 unsharp 生效、确定性、代价与图像差异幅度，并落盘 4K JPEG 供代理复核。"""
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline')
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import pixel_io  # noqa: E402
import r_ledger  # noqa: E402
import score01_bench as SB  # noqa: E402

run = Path(r'D:\models\opt10')
base = Path(r'D:\models\opt01\npy')
JPEG = {'quality': 98, 'subsampling': 0, 'optimize': False, 'progressive': False}
frozen_mode.apply_frozen_mode()
led = r_ledger.Ledger.from_file(run / 'ledger_opt10.json')
led.acquire_lock('OPT-10 verify')
out = {}
try:
    led.open_attempt('opt10_verify_%d' % os.getpid(), run)
    rA = SB.load_runner(Path(r'D:\models\opt01\model_dir'))
    rB = SB.load_runner(run / 'variant' / 'model_dir')
    x = SB.load_input('c1_512')
    # 512 侧：A/B 各 3 次（看代价与确定性）
    res = {}
    for tag, r in (('A_opt01', rA), ('B_opt10', rB)):
        calls = []
        ys = []
        for i in range(3):
            led.reserve('top', 'opt10v_p%d_%s_%d' % (os.getpid(), tag, i + 1))
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                y = r.infer(x, prompt='').detach().float().cpu()
            torch.cuda.synchronize()
            calls.append(round((time.perf_counter() - t) * 1000, 1))
            ys.append(y)
        d = (ys[0] - ys[2]).abs().max()
        ref = torch.from_numpy(np.load(base / 'c1_512_candidate.npy', allow_pickle=False)).float()
        res[tag] = {'calls_ms': calls, 'repeat_max_abs': float(d),
                    'vs_opt01_max': float((ref - ys[0]).abs().max()),
                    'vs_opt01_mean': float((ref - ys[0]).abs().mean())}
        print(tag, json.dumps(res[tag]), flush=True)
    out.update(res)
    out['unsharp_cost_ms'] = round(res['B_opt10']['calls_ms'][-1] - res['A_opt01']['calls_ms'][-1], 2)
    # 4K：cases 1,5,25,88 → 存 JPEG 供代理复核
    outdir = run / 'npy' / 'full4k'
    outdir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as _I
    for case in (1, 5, 25, 88):
        im = _I.open(SB.SRC / ('case%d.jpg' % case)).convert('RGB')
        arr = np.asarray(im)
        xx = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        led.reserve('top', 'opt10v_p%d_case%d_full' % (os.getpid(), case), is_4k=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            y = rB.infer(xx, prompt='').detach().float().cpu()
        ms = round((time.perf_counter() - t) * 1000, 1)
        u8 = pixel_io.to_u8(y)
        p = outdir / ('case%d.jpg' % case)
        _I.fromarray(u8).save(p, format='JPEG', **JPEG)
        out.setdefault('full4k', {})[case] = {'wall_ms': ms,
                                              'finite': bool(torch.isfinite(y).all())}
        print('4K case%d %.0fms' % (case, ms), flush=True)
    out['budget'] = led._read()['used']
    (run / 'evidence' / 'opt10_verify.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                        encoding='utf-8')
    print('budget', out['budget'])
finally:
    try:
        led.close_attempt('OK')
    except Exception:
        pass
    led.release_lock()
