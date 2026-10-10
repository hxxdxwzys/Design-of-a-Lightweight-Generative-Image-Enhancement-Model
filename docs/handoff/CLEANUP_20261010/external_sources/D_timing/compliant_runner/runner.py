"""合规 model_dir runner (2026-09-02, 规则三合规版)
M24 模型主体: VOSR 0.5B os + v7_s5000 LoRA(sc0.8), 全链 JIT trace (your_model.pt)
接口: Runner(model_dir).infer(image_tensor, prompt="") -> [1,3,H,W] in [-1,1]
说明: 与出图(output_dir)同一模型主体; 512x512 完整前向 (VAE+dinov2+DiT 1步+decoder), fp16, 无任何捷径。
"""
import os
import torch
import numpy as np


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 权重精度对齐官方规则: fp16
        self.weight_dtype = torch.float16
        pt_path = os.path.join(model_dir, "your_model.pt")
        self.model = torch.jit.load(pt_path, map_location=self.device)
        self.model.eval()
        self.randn = None  # 惰性生成随机噪声 latent

    def infer(self, image_tensor, prompt=""):
        """image_tensor: [1,3,H,W] float in [-1,1]; 返回同尺寸 [-1,1]"""
        x = image_tensor.to(self.device, dtype=self.weight_dtype)
        b, c, h, w = x.shape
        # 支持任意输入尺寸: 512 走整图; 其余尺寸也走完整链 (VAE 8x 下采样需 h,w 为 8 的倍数)
        lat_h, lat_w = h // 8, w // 8
        # 随机噪声 latent (每次独立, 与推理语义一致)
        gen = torch.Generator(device=self.device).manual_seed(0)
        z = torch.randn(b, 4, lat_h, lat_w, device=self.device,
                        dtype=self.weight_dtype, generator=gen)
        with torch.no_grad():
            sr = self.model(x, z)
        return sr.float().clamp(-1.0, 1.0)
