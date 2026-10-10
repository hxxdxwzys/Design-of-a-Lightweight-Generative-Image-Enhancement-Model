# -*- coding: utf-8 -*-
"""FP32 候选组件门槛唯一失败项（dino_512 max=5.798e-4）的最小成因探针（CPU 编排、GPU 6 次组件前向）。

问题：全 FP32 导出件与同配方 eager 参考在 DINO 特征上差 max=5.798e-4（参考量级 335 ⇒ 相对 1.79e-6），
而编码器/一步 DiT/解码/wavelet 在 fp32 下均为 0.0；fp16 候选（A2）同一项为 0.0。
本探针只回答：差异是否"trace 图 vs eager"在**同一进程内**复现，以及是否随 SDPA 后端固定而消失。
"""
from __future__ import annotations

import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402

RUN = Path(r'C:\models\exports\DELIV02FP32_20260918_113449')
EV = HERE / 'evidence' / 'DELIV02FP32_20260918_113449'
LEDGER = Path(r'C:\models\exports\DELIV02_20260918_073810\ledger.json')
MDIR = RUN / 'model_dir_fp32'


def stats(a, b):
    d = (a - b).abs()
    return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
            'n_gt_1e-5': int((d > 1e-5).sum()), 'n_gt_1e-4': int((d > 1e-4).sum()),
            'ref_absmax': float(b.abs().max()), 'check_absmax': float(a.abs().max())}


def sdp_state():
    b = torch.backends.cuda
    return {'flash': bool(b.flash_sdp_enabled()), 'mem_efficient': bool(b.mem_efficient_sdp_enabled()),
            'math': bool(b.math_sdp_enabled()), 'cudnn_sdp': bool(b.cudnn_sdp_enabled())}


def main():
    flags = frozen_mode.apply_frozen_mode()
    dev = torch.device('cuda')
    out = {'schema': 'DELIV02-fp32-dino-probe/1', 'flags': flags,
           'default_sdp': sdp_state(), 'tolerances': {'max_abs': 1e-5, 'mean_abs': 1e-6}}
    led = r_ledger.Ledger.from_file(LEDGER)
    led.acquire_lock('FP32 dino probe')
    led.open_attempt('fp32_dino_probe', RUN)
    try:
        import core as PBcore
        led.reserve('component', 'probe_load_assets')
        cfg = PBcore.load_config()
        assets = PBcore.Assets(cfg, dev)
        from chain import DinoFeat
        eager = DinoFeat(assets.venc.to(torch.float32), size=int(cfg['expected_dinov2_size']),
                         layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                         n_blocks=len(assets.venc.blocks)).to(dev).eval()
        model = torch.jit.load(str(MDIR / 'your_model.pt'), map_location=dev).eval()
        x = torch.from_numpy(np.load(EV / 'component_refs' / 'in_dino_512.npy',
                                     allow_pickle=False)).to(dev).to(torch.float32)
        out['input_sha256'] = __import__('hashlib').sha256(
            x.detach().cpu().numpy().tobytes()).hexdigest()

        def run_pair(tag):
            with torch.no_grad():
                led.reserve('component', 'probe_eager_%s' % tag)
                e = eager(x)
                led.reserve('component', 'probe_traced_%s' % tag)
                t = model.features(x)
                led.reserve('component', 'probe_eager2_%s' % tag)
                e2 = eager(x)
                led.reserve('component', 'probe_traced2_%s' % tag)
                t2 = model.features(x)
            rec = {'eager_vs_traced': stats(t, e), 'eager_self': stats(e2, e),
                   'traced_self': stats(t2, t), 'sdp': sdp_state()}
            out[tag] = rec
            out.setdefault('seq', []).append(tag)
            print('[probe] %-14s eager-vs-traced max=%.4e mean=%.4e | self e=%.2e t=%.2e'
                  % (tag, rec['eager_vs_traced']['max_abs'], rec['eager_vs_traced']['mean_abs'],
                     rec['eager_self']['max_abs'], rec['traced_self']['max_abs']), flush=True)
            return rec

        run_pair('default_backends')
        # 固定为 math-only：eager 与 trace 图都必须走同一实现
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        run_pair('math_only')
        # 复原，避免影响后续进程使用的全局状态（本进程结束即退出）
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        (EV / 'fp32_dino_probe.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                encoding='utf-8')
        print('[probe] written', EV / 'fp32_dino_probe.json', flush=True)
    finally:
        try:
            led.close_attempt('OK' if (EV / 'fp32_dino_probe.json').is_file() else 'FAIL')
        except Exception:
            pass
        led.release_lock()


if __name__ == '__main__':
    main()
