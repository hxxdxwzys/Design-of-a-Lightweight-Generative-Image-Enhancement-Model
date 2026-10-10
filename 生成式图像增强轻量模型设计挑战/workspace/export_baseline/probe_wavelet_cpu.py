# -*- coding: utf-8 -*-
"""CPU 探针：wavelet 后处理张量化可行性（不加载模型、不用 GPU）。

回答三个问题（EXP-05 预登记前的事实核验）：
  Q1 tensor_to_pil 的浮点->uint8 舍入规则（round-half-up 还是截断）？
  Q2 同尺寸 LANCZOS resize 是否为恒等？
  Q3 张量化高斯模糊（sigma=5, ksize=41, reflect101）与 cv2 的差异量级？
"""
import sys
from pathlib import Path

import numpy as np
import torch

PB = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\pipeline_baseline')
sys.path.insert(0, str(PB))
import core as PBcore  # noqa: E402

from PIL import Image  # noqa: E402
import cv2  # noqa: E402
from torchvision.transforms.functional import to_pil_image  # noqa: E402


def q_round_half_up(t):
    return torch.clamp((t * 255.0) + 0.5, 0, 255).to(torch.uint8)


def q_trunc(t):
    return torch.clamp(t * 255.0, 0, 255).to(torch.uint8)


def main():
    torch.manual_seed(0)
    # Q1: 构造含 .5 边界的浮点张量，比较两种候选规则
    vals = torch.tensor([[[[0.0, 0.5 / 255, 1.0 / 255, 1.5 / 255, 2.5 / 255, 254.5 / 255]]]])
    t01 = vals.clone()
    pil = to_pil_image(t01[0])
    got = np.array(pil).astype(np.int64)[0, 0]
    print('Q1 expected round_half_up =', q_round_half_up(torch.clamp(t01, 0, 1))[0, 0, 0].tolist())
    print('Q1 expected trunc         =', q_trunc(torch.clamp(t01, 0, 1))[0, 0, 0].tolist())
    print('Q1 actual to_pil_image    =', got.tolist())

    # Q2: 同尺寸 LANCZOS resize 是否恒等
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    b = np.array(im.resize(im.size, Image.LANCZOS))
    print('Q2 same_size_lanczos_identical =', bool(np.array_equal(a, b)),
          ' max_abs_uint8 =', int(np.abs(a.astype(int) - b.astype(int)).max()))

    # Q3: 张量化高斯模糊 vs cv2（float32, sigma=5, ksize=(0,0)）
    sigma = 5.0
    ks = int(round(sigma * 4 * 2 + 1)) | 1
    xs = torch.arange(ks, dtype=torch.float64) - (ks - 1) / 2.0
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    k1f = k1.to(torch.float32)
    print('Q3 ksize =', ks, ' kernel_sum =', float(k1.sum()))

    def blur_torch(x_np):
        t = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        w1 = k1f.view(1, 1, 1, ks).expand(3, 1, 1, ks).contiguous()
        w2 = k1f.view(1, 1, ks, 1).expand(3, 1, ks, 1).contiguous()
        t = torch.nn.functional.pad(t, (ks // 2, ks // 2, 0, 0), mode='reflect')
        t = torch.nn.functional.conv2d(t, w1, groups=3)
        t = torch.nn.functional.pad(t, (0, 0, ks // 2, ks // 2), mode='reflect')
        t = torch.nn.functional.conv2d(t, w2, groups=3)
        return t[0].permute(1, 2, 0).numpy()

    for name, arr in (('random', rng.random((96, 80, 3)).astype(np.float32)),
                      ('gray', np.full((64, 64, 3), 0.5, dtype=np.float32)),
                      ('gradient', np.linspace(0, 1, 96 * 80 * 3, dtype=np.float32).reshape(96, 80, 3))):
        ref = cv2.GaussianBlur(arr, (0, 0), sigma)
        got = blur_torch(arr)
        d = np.abs(ref - got)
        print('Q3 %-9s cv2_vs_torch max_abs=%.3e mean_abs=%.3e' % (name, d.max(), d.mean()))

    # Q4: 端到端对照（参考张量化实现 vs PBcore.wavelet_color_fix_tensor）
    def wavelet_torch(target, source, sigma=5.0):
        tgt_u8 = q_round_half_up(torch.clamp(target, -1, 1) * 0.5 + 0.5)   # (待 Q1 结论确认)
        src_u8 = q_round_half_up(torch.clamp(source, -1, 1) * 0.5 + 0.5)
        tn = tgt_u8.float() / 255.0
        sn = src_u8.float() / 255.0
        t_low = torch.from_numpy(blur_torch(tn[0].permute(1, 2, 0).numpy())).permute(2, 0, 1).unsqueeze(0)
        s_low = torch.from_numpy(blur_torch(sn[0].permute(1, 2, 0).numpy())).permute(2, 0, 1).unsqueeze(0)
        res = torch.clamp(s_low + (tn - t_low), 0, 1) * 255.0
        res_u8 = torch.clamp(torch.floor(res), 0, 255).to(torch.uint8)
        return (res_u8.float() / 255.0) * 2.0 - 1.0

    for name in ('random', 'gray', 'gradient'):
        if name == 'random':
            tgt_t = torch.from_numpy(rng.random((1, 3, 96, 80)).astype(np.float32)) * 2 - 1
            src_t = torch.from_numpy(rng.random((1, 3, 96, 80)).astype(np.float32)) * 2 - 1
        elif name == 'gray':
            tgt_t = torch.full((1, 3, 64, 64), -0.2, dtype=torch.float32)
            src_t = torch.full((1, 3, 64, 64), 0.3, dtype=torch.float32)
        else:
            tgt_t = (torch.linspace(-1, 1, 96 * 80).reshape(1, 1, 96, 80).repeat(1, 3, 1, 1))
            src_t = (torch.linspace(1, -1, 96 * 80).reshape(1, 1, 96, 80).repeat(1, 3, 1, 1))
        ref = PBcore.wavelet_color_fix_tensor(tgt_t, src_t, sigma=5)
        got = wavelet_torch(tgt_t, src_t, sigma=5)
        d = (ref - got).abs()
        print('Q4 %-9s e2e max_abs=%.3e mean_abs=%.3e  pixel_equal=%s'
              % (name, float(d.max()), float(d.mean()),
                 bool(torch.equal(Q := q_round_half_up((ref + 1) / 2), q_round_half_up((got + 1) / 2)))))


if __name__ == '__main__':
    main()
