# -*- coding: utf-8 -*-
"""EXP-R05 合并对照探针：独立进程、同精度、同输入，只改变"是否合并 LoRA"。

角色模式（驱动脚本以子进程方式各跑一次；父权重独立加载，不在 half 后的实例上做 float）：
  python r_merge_probe.py role <peft|merged> <out_npy>
驱动模式（worker 内调用；组件前向额度由 worker 先扣）：
  python r_merge_probe.py drive <exp_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

PB = Path(__file__).resolve().parent.parent / 'pipeline_baseline'
if str(PB) not in sys.path:
    sys.path.insert(0, str(PB))


def _inputs():
    import torch
    g = torch.Generator(device='cpu')
    g.manual_seed(42)
    latent = torch.randn(1, 8, 64, 64, generator=g, dtype=torch.float32) * 0.1
    feats = torch.randn(1, 1024, 768, generator=g, dtype=torch.float32) * 0.1
    t = torch.full((1,), 1.0)
    r = torch.full((1,), 0.0)
    return latent, feats, t, r


def role(which, out_path):
    import torch
    import frozen_mode
    frozen_mode.apply_frozen_mode()      # DELIV-01 P0：子进程自己设置并记录冻结开关
    import core as PBcore
    dev = torch.device('cuda')
    cfg = PBcore.load_config()
    PBcore.ensure_offline_env(cfg)
    PBcore.ensure_vosr_import_path(cfg)   # 必须：build_dit 依赖 VOSR 的 models 包（EXP-R05 修复点）
    from safetensors.torch import load_file
    base_sd = load_file(str(cfg['base_weights']))
    args = PBcore.build_args_ns(cfg)
    model, _ = PBcore.build_dit(cfg, args, base_state=base_sd)
    del base_sd
    payload = PBcore.load_lora_payload(cfg['lora_path'])
    model, _ = PBcore.apply_lora(model, payload, cfg['lora_scale'])
    model = model.to(dev).eval()
    if which == 'merged':
        model = model.merge_and_unload().to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    latent, feats, t, r = _inputs()
    with torch.no_grad():
        u = model.forward_flexible(latent.to(dev), t.to(dev), r.to(dev), [feats.to(dev)])
    np.save(out_path, u.detach().cpu().float().numpy())
    print(json.dumps({'role': which, 'shape': list(u.shape), 'dtype': str(u.dtype)}))


def drive(exp_dir):
    import hashlib
    import subprocess
    exp = Path(exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    outs = {}
    for which in ('peft', 'merged'):
        p = exp / ('merge_%s.npy' % which)
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'role', which,
                             str(p)], capture_output=True, text=True, timeout=1800)
        if cp.returncode != 0:
            raise RuntimeError('merge probe %s 失败：%s' % (which, (cp.stderr or '')[-500:]))
        outs[which] = p
    a = np.load(outs['peft'], allow_pickle=False)
    b = np.load(outs['merged'], allow_pickle=False)
    rec = {'schema': 'EXPORT01R-merge-probe/1',
           'design': '两个独立进程各自从父权重加载 fp32；peft 未合并 vs merged 已合并；同精度同输入',
           'input_protocol': 'latent(1,8,64,64)/feats(1,1024,768)/t=1/r=0，CPU Generator(seed=42) 重建',
           'peft_npy_sha256': hashlib.sha256(outs['peft'].read_bytes()).hexdigest(),
           'merged_npy_sha256': hashlib.sha256(outs['merged'].read_bytes()).hexdigest(),
           'max_abs': float(np.abs(a - b).max()), 'mean_abs': float(np.abs(a - b).mean()),
           'shape': list(a.shape),
           'finite': bool(np.isfinite(a).all() and np.isfinite(b).all())}
    (exp / 'merge_probe.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                          encoding='utf-8')
    print(json.dumps({'max_abs': rec['max_abs'], 'mean_abs': rec['mean_abs']}))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'role':
        role(sys.argv[2], sys.argv[3])
    else:
        drive(sys.argv[2])
