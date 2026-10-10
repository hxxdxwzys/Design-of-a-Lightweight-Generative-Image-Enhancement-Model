"""轻量一步潜扩散模型 (同分辨率真实图像增强 LQ→HQ)

组成：
  - VAE: SD2.1 AutoencoderKL (冻结，复用其高质量潜空间)
  - UNet: 自研小型 UNet2DModel (无 cross-attn / 无文本条件)，可训
  - Scheduler: DDPMScheduler，单步 t=999

前向 (与 OSEDiff 同范式，但去掉 RAM/CLIP/大UNet)：
  lq[-1,1] -> VAE.encode -> lq_latent
  -> UNet(lq_latent, t=999) 预测噪声
  -> scheduler.step -> x0_latent
  -> VAE.decode -> out[-1,1]

图像自身作输入条件（lq_latent 即 UNet 的 sample），无需文本。
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler


class LightLatentDiff(nn.Module):
    def __init__(self, sd_path, unet_ch=(64, 128, 256, 512), layers_per_block=2,
                 num_train_timesteps=1000, t=500):
        super().__init__()
        # --- VAE (冻结) ---
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling = self.vae.config.scaling_factor

        # --- 小型 UNet (可训, 无 cross-attn) ---
        n = len(unet_ch)
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=4, out_channels=4,
            layers_per_block=layers_per_block,
            block_out_channels=unet_ch,
            down_block_types=("DownBlock2D",) * n,
            up_block_types=("UpBlock2D",) * n,
        )

        # --- 调度器 (仅用 alphas_cumprod) ---
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            beta_start=0.00085, beta_end=0.012,
        )
        self.t = t
        # 固定 t 的 alpha_cumprod，用于显式 x0 预测公式 (可导、可 trace)
        self.register_buffer("alpha_t", self.scheduler.alphas_cumprod[t].clone())

    @torch.no_grad()
    def encode(self, img):
        """img[-1,1] -> latent (已乘 scaling)"""
        return self.vae.encode(img).latent_dist.sample() * self.scaling

    def decode(self, latent):
        """latent (含scaling) -> img[-1,1]，保留梯度(供像素级损失反传到UNet)"""
        return self.vae.decode(latent / self.scaling).sample.clamp(-1, 1)

    def forward(self, lq, return_x0=False):
        """
        lq: [B,3,H,W] in [-1,1]
        1-step: VAE.encode(lq)->latent; UNet 预测 epsilon; 显式 x0 = (lat - sqrt(1-a)*eps)/sqrt(a)
        return: out [B,3,H,W] in [-1,1]
        """
        lq_lat = self.encode(lq)
        t = torch.full((lq.size(0),), self.t, device=lq.device, dtype=torch.long)
        pred_noise = self.unet(lq_lat, t).sample
        a = self.alpha_t
        x0 = (lq_lat - (1.0 - a).sqrt() * pred_noise) / a.sqrt()
        out = self.decode(x0)
        if return_x0:
            return out, x0, lq_lat
        return out

    @torch.no_grad()
    def infer(self, lq):
        """推理接口 (与官方 runner 一致: [1,3,H,W]∈[-1,1] -> [1,3,H,W]∈[-1,1])"""
        self.eval()
        return self.forward(lq)

    def trainable_params(self):
        return list(self.unet.parameters())

    def param_count(self):
        return sum(p.numel() for p in self.unet.parameters())
