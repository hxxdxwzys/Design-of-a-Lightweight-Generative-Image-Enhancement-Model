# -*- coding: utf-8 -*-
"""合规 runner 端到端验证 (2026-09-03): Runner.infer 512 推理 + 与 vosr_lora_infer 输出统计对比
"""
import importlib.util
import sys
import time

import numpy as np
import torch
from PIL import Image

MD = r'D:/models/final_compliant/model_dir'


def load_runner():
    spec = importlib.util.spec_from_file_location('compliant_runner', MD + '/runner.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['compliant_runner'] = mod
    spec.loader.exec_module(mod)
    return mod.Runner


def main():
    img = Image.open(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集/case1_lq.jpg').convert('RGB')
    img512 = img.resize((512, 512), Image.BICUBIC)
    x = (np.asarray(img512).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    xt = torch.from_numpy(x).unsqueeze(0)

    t0 = time.time()
    Runner = load_runner()
    r = Runner(MD)
    load_t = time.time() - t0
    print(f'[runner] 加载耗时 {load_t:.1f}s', flush=True)

    # warmup + 计时
    with torch.no_grad():
        out = r.infer(xt)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(3):
        with torch.no_grad():
            r.infer(xt)
    torch.cuda.synchronize()
    dt = (time.time() - t1) / 3
    o = out[0]
    print(f'[infer] shape={tuple(o.shape)} dtype={o.dtype} '
          f'std={o.float().std().item():.3f} mean={o.float().mean().item():.3f} '
          f'min={o.float().min().item():.2f} max={o.float().max().item():.2f}', flush=True)
    print(f'[speed] 512 全链 {dt * 1000:.1f} ms (本地 4060, Runner.infer 含 randn/IO)', flush=True)
    assert o.shape == (3, 512, 512), '输出形状错误'
    assert o.float().std().item() > 0.05, '输出异常 (std 过小)'
    assert float(o.float().min()) >= -1.0 and float(o.float().max()) <= 1.0, '输出超出 [-1,1]'
    # 保存目测
    out_png = (o.float().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((np.clip(out_png, 0, 1) * 255).astype(np.uint8)).save(r'D:/tmp_timing/runner_out_512.png')
    print('[save] runner_out_512.png (目测用)', flush=True)
    print('RUNNER_E2E_PASS', flush=True)


if __name__ == '__main__':
    main()
