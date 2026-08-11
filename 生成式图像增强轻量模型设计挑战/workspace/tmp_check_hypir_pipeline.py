"""用 HYPIR 训练管线 (RealESRGANDataset + RealESRGANBatchTransform, E4 配置)
在 3 张官方 GT (4K) 上生成合成 LQ, 验证统计与官方 LQ 对齐。

官方 LQ 基线: clipiqa 0.251 / musiq 24.09 / maniqa 0.302 / niqe 7.67
用法: python tmp_check_hypir_pipeline.py
"""
import sys
import math
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, r"D:/models/hypir_code/HYPIR-main/HYPIR-main")
from omegaconf import OmegaConf
from HYPIR.utils.common import instantiate_from_config
from HYPIR.utils.degradation import circular_lowpass_kernel, random_mixed_kernels

VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")
OUT = Path(r"D:/models/hypir/train_pipe_check")
OUT.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(r"D:/models/hypir_code/HYPIR-main/HYPIR-main/configs/sd2_train_1x_v2nos.yaml")
# 4K 检查脚本: 关闭 queue (训练时 512 patch 用 32; 4K 混合尺寸会 shape 冲突)
cfg.data_config.train.batch_transform.params.queue_size = 0
ds = instantiate_from_config(cfg.data_config.train.dataset)
bt = instantiate_from_config(cfg.data_config.train.batch_transform)


def gen_kernel(ds, sigma, prob, kprob):
    kernel_size = random.choice(ds.kernel_range)
    if random.random() < prob:
        if kernel_size < 13:
            omega_c = random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            ds.kernel_list, kprob, kernel_size, sigma, sigma,
            [-math.pi, math.pi], ds.betag_range, ds.betap_range, noise_range=None)
    pad = (21 - kernel_size) // 2
    return np.pad(kernel, ((pad, pad), (pad, pad)))


for case in (1, 2, 3):
    gt_p = VAL / f"case{case}_gt.png"
    if not gt_p.exists():
        gt_p = VAL / f"case{case}_gt.jpg"
    img = Image.open(gt_p).convert("RGB")
    hq = torch.from_numpy(np.asarray(img)[..., ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).float()  # BGR [0,1]
    k1 = torch.FloatTensor(gen_kernel(ds, ds.blur_sigma, ds.sinc_prob, ds.kernel_prob))
    k2 = torch.FloatTensor(gen_kernel(ds, ds.blur_sigma2, ds.sinc_prob2, ds.kernel_prob2))
    if random.random() < ds.final_sinc_prob:
        ksz = random.choice(ds.kernel_range)
        sk = circular_lowpass_kernel(random.uniform(np.pi / 3, np.pi), ksz, pad_to=21)
        sinc = torch.FloatTensor(sk)
    else:
        sinc = ds.pulse_tensor
    batch = {"hq": hq.unsqueeze(0), "kernel1": k1.unsqueeze(0),
             "kernel2": k2.unsqueeze(0), "sinc_kernel": sinc.unsqueeze(0), "txt": [""]}
    out = bt(batch)
    lq = out["LQ"][0].permute(1, 2, 0)
    rgb = lq.numpy()[..., ::-1]
    Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(OUT / f"case{case}.png")
    print(f"case{case}: saved synthetic LQ ({hq.shape[-2]}x{hq.shape[-1]})")
print("done -> eval with: python eval.py --pred-dir D:/models/hypir/train_pipe_check --pred-pattern '*.png' --metrics clipiqa,musiq,maniqa,niqe")
